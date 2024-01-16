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
    @testset "lift_getfield_and_others $ex" for (ex, target) in Any[
        (ReturnNode(5), ReturnNode(5)),
        (
            Expr(:call, getfield, SSAValue(1), 5),
            Expr(:call, lgetfield, SSAValue(1), Val(5)),
        ),
        (
            Expr(:call, GlobalRef(Core, :getfield), SSAValue(1), 5),
            Expr(:call, lgetfield, SSAValue(1), Val(5)),
        ),
        (
            Expr(:call, QuoteNode(getfield), SSAValue(1), 5),
            Expr(:call, lgetfield, SSAValue(1), Val(5)),
        ),
        (
            Expr(:call, getfield, SSAValue(1), SSAValue(2)),
            Expr(:call, getfield, SSAValue(1), SSAValue(2)),
        ),
        (
            Expr(:call, getfield, SSAValue(1), QuoteNode(:x)),
            Expr(:call, lgetfield, SSAValue(1), Val(:x)),
        ),
        (
            Expr(:call, sin, SSAValue(1)),
            Expr(:call, sin, SSAValue(1)),
        ),
    ]
        @test Taped.lift_getfield_and_others(ex) == target
    end
end
