@testset "ir_normalisation" begin
    @testset "invoke_to_call" begin
        mi = which(Tuple{typeof(sin), Float64}).specializations
        invoke_inst = Expr(:invoke, mi, sin, 5.0)
        call_inst = Taped.invoke_to_call(invoke_inst)
        @test call_inst.head == :call
        @test call_inst.args == invoke_inst.args[2:end]
    end
    @testset "foreigncall_to_call" begin
        foreigncall = Expr(
            :foreigncall,
            :(:jl_array_isassigned),
            Int32,
            svec(Any, UInt64),
            0,
            :(:ccall),
            Argument(2),
            0x0000000000000001,
            0x0000000000000001,
        )
        sp_map = Dict{Symbol, CC.VarState}()
        call = Taped.foreigncall_to_call(foreigncall, sp_map)
        @test Meta.isexpr(call, :call)
        @test call.args[1] == Taped._foreigncall_
    end
    @testset "new_to_call" begin
        new_ex = Expr(:new, GlobalRef(Taped, :Foo), SSAValue(1), :hi)
        call_ex = Taped.new_to_call(new_ex)
        @test Meta.isexpr(call_ex, :call)
        @test call_ex.args[1] == Taped._new_
        @test call_ex.args[2:end] == new_ex.args
    end
    @testset "intrinsic_to_function" begin
        @testset "GlobalRef" begin
            intrinsic_ex = Expr(:call, GlobalRef(Core.Intrinsics, :abs_float), SSAValue(1))
            wrapper_ex = Taped.intrinsic_to_function(intrinsic_ex)
            @test wrapper_ex.args[1] == Taped.IntrinsicsWrappers.abs_float
        end
        @testset "IntrinsicFunction" begin
            intrinsic_ex = Expr(:call, Core.Intrinsics.abs_float, SSAValue(1))
            wrapper_ex = Taped.intrinsic_to_function(intrinsic_ex)
            @test wrapper_ex.args[1] == Taped.IntrinsicsWrappers.abs_float
        end
        @testset "cglobal" begin
            cglobal_ex = Expr(:call, cglobal, :jl_uv_stdout, Ptr{Cvoid})
            wrapper_ex = Taped.intrinsic_to_function(cglobal_ex)
            @test wrapper_ex.args[1] == Taped.IntrinsicsWrappers.__cglobal
        end
    end
    @testset "rebind_phi_nodes!" begin
        ir = Taped.ircode(
            Any[
                Expr(:call, println, "1"),
                PhiNode(Int32[1, 3], Any[SSAValue(1), Argument(2)]),
                PhiNode(Int32[1, 3], Any[SSAValue(2), false]),
                Expr(:call, sin, SSAValue(1)),
                Expr(:call, +, SSAValue(4), SSAValue(2)),
                GotoIfNot(SSAValue(3), 8),
                Expr(:call, println, "hmm"),
                ReturnNode(SSAValue(5)),
            ],
            Any[Tuple{}, Float64],
        )
        rebound_ir = Taped.rebind_phi_nodes!(CC.copy(ir))

        # Check first rebind is correct.
        first_rebind = rebound_ir.stmts.inst[4]
        @test Meta.isexpr(first_rebind, :call)
        @test first_rebind.args[1] == Taped.__rebind
        @test first_rebind.args[2] == SSAValue(2)

        # Check second rebind is correct.
        second_rebind = rebound_ir.stmts.inst[5]
        @test Meta.isexpr(second_rebind, :call)
        @test second_rebind.args[1] == Taped.__rebind
        @test second_rebind.args[2] == SSAValue(3)

        # Check that the reference to the first PhiNode in the second PhiNode is replaced.
        second_phi_node = rebound_ir.stmts.inst[3]
        @test second_phi_node isa PhiNode
        @test second_phi_node.values[1] == SSAValue(4)

        # Check that the reference to the first PhiNode in the `+` call is replaced.
        plus_call = rebound_ir.stmts.inst[7]
        @test Meta.isexpr(plus_call, :call)
        @test plus_call.args == Any[+, SSAValue(6), SSAValue(4)]

        # Check that the reference to thesecond PhiNode in the `GotoIfNot` node is replaced.
        gotoifnot = rebound_ir.stmts.inst[8]
        @test gotoifnot isa CC.GotoIfNot
        @test gotoifnot.cond == SSAValue(5)
    end
    @testset "rebind_multiple_usage!" begin
        @testset "single-line case" begin
            ir = Taped.ircode(
                Any[
                    Expr(:call, GlobalRef(Base, :(+)), Argument(1), Argument(1)),
                ],
                Any[Tuple{}, Float64],
            )
            new_ir = Taped.rebind_multiple_usage!(CC.copy(ir))
            @test new_ir.stmts.inst == Any[
                Expr(:call, Taped.__rebind, Argument(1)),
                Expr(:call, GlobalRef(Base, :(+)), Argument(1), SSAValue(1)),
            ]
        end
        @testset "function which calls itself" begin
            ir = Taped.ircode(
                Any[Expr(:call, Argument(1), Argument(1)),], Any[Tuple{}, Float64],
            )
            new_ir = Taped.rebind_multiple_usage!(CC.copy(ir))
            @test new_ir.stmts.inst == Any[
                Expr(:call, Taped.__rebind, Argument(1)),
                Expr(:call, Argument(1), SSAValue(1)),
            ]
        end
        @testset "Lots of uses of SSAValue" begin
            ir = Taped.ircode(
                Any[
                    Expr(:call, sin, Argument(2)),
                    Expr(:call, Argument(1), SSAValue(1), SSAValue(1), SSAValue(1)),
                ],
                Any[Tuple{}, Float64],
            )
            new_ir = Taped.rebind_multiple_usage!(CC.copy(ir))
            @test new_ir.stmts.inst == Any[
                Expr(:call, sin, Argument(2)),
                Expr(:call, Taped.__rebind, SSAValue(1)),
                Expr(:call, Taped.__rebind, SSAValue(1)),
                Expr(:call, Argument(1), SSAValue(1), SSAValue(2), SSAValue(3)),
            ]
        end
        @testset "Spaced out usage" begin
            ir = Taped.ircode(
                Any[
                    Expr(:call, sin, Argument(2)),
                    Expr(:call, Argument(1), SSAValue(1), Argument(3), SSAValue(1)),
                ],
                Any[Tuple{}, Float64],
            )
            new_ir = Taped.rebind_multiple_usage!(CC.copy(ir))
            @test new_ir.stmts.inst == Any[
                Expr(:call, sin, Argument(2)),
                Expr(:call, Taped.__rebind, SSAValue(1)),
                Expr(:call, Argument(1), SSAValue(1), Argument(3), SSAValue(2)),
            ]
        end
    end
end
