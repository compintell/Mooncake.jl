@testset "ir_utils" begin
    @testset "ircode $(typeof(fargs))" for fargs in Any[
        (sin, 5.0), (cos, 1.0),
    ]
        # Construct a vector of instructions from known function.
        f, args... = fargs
        insts = only(code_typed(f, Tuple{map(Core.Typeof, args)...}))[1].code
    
        # Use Taped.ircode to build an `IRCode`.
        argtypes = Any[map(Core.Typeof, fargs)...]
        ir = Taped.ircode(insts, argtypes)

        # Check the validity of the `IRCode`, and that an OpaqueClosure constructed using it
        # gives the same answer as the original function.
        @test length(ir.stmts.inst) == length(insts)
        @test Core.OpaqueClosure(ir; do_compile=true)(args...) == f(args...)
    end
    @testset "infer_ir!" begin

        # Generate IR without any types. 
        ir = Taped.ircode(
            Any[
                Expr(:call, GlobalRef(Base, :sin), Argument(2)),
                Expr(:call, cos, SSAValue(1)),
                ReturnNode(SSAValue(2)),
            ],
            Any[Tuple{}, Float64],
        )

        # Run inference and check that the types are as expected.
        ir = Taped.infer_ir!(ir)
        @test ir.stmts.type[1] == Float64
        @test ir.stmts.type[2] == Float64

        # Check that the ir is runable.
        @test Core.OpaqueClosure(ir)(5.0) == cos(sin(5.0))
    end
    @testset "replace_all_uses_with!" begin

        # `replace_all_uses_with!` is just a lightweight wrapper around `replace_uses_with`,
        # so we just test that carefully.
        @testset "replace_uses_with $val" for (val, target) in Any[
            (5.0, 5.0),
            (5, 5),
            (Expr(:call, sin, SSAValue(1)), Expr(:call, sin, SSAValue(2))),
            (Expr(:call, sin, SSAValue(3)), Expr(:call, sin, SSAValue(3))),
            (GotoNode(1), GotoNode(1)),
            (GotoIfNot(false, 5), GotoIfNot(false, 5)),
            (GotoIfNot(SSAValue(1), 3), GotoIfNot(SSAValue(2), 3)),
            (GotoIfNot(SSAValue(3), 3), GotoIfNot(SSAValue(3), 3)),
            (
                PhiNode(Int32[1, 2, 3], Any[5, SSAValue(1), SSAValue(3)]),
                PhiNode(Int32[1, 2, 3], Any[5, SSAValue(2), SSAValue(3)]),
            ),
            (PiNode(SSAValue(1), Float64), PiNode(SSAValue(2), Float64)),
            (PiNode(SSAValue(3), Float64), PiNode(SSAValue(3), Float64)),
            (PiNode(Argument(1), Float64), PiNode(Argument(1), Float64)),
            (QuoteNode(:a_quote), QuoteNode(:a_quote)),
            (ReturnNode(5), ReturnNode(5)),
            (ReturnNode(SSAValue(1)), ReturnNode(SSAValue(2))),
            (ReturnNode(SSAValue(3)), ReturnNode(SSAValue(3))),
            (ReturnNode(), ReturnNode()),
        ]
            @test Taped.replace_uses_with(val, SSAValue(1), SSAValue(2)) == target
        end
        @testset "PhiNode with undefined" begin
            vals_with_undef_1 = Vector{Any}(undef, 2)
            vals_with_undef_1[2] = SSAValue(1)
            val = PhiNode(Int32[1, 2], vals_with_undef_1)
            result = Taped.replace_uses_with(val, SSAValue(1), SSAValue(2))
            @test result.values[2] == SSAValue(2)
            @test !isassigned(result.values, 1)
        end
    end
end
