@testset "ir_normalisation" begin
    @testset "ensure_single_argument_usage_per_call!" begin
        ir = Base.code_ircode(x -> x * x, Tuple{Float64})
        ir = CC.IRCode()

    end
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
    @testset "new_expr_to_call_expr" begin
        new_ex = Expr(:new, GlobalRef(Taped, :Foo), SSAValue(1), :hi)
        call_ex = Taped.new_expr_to_call_expr(new_ex)
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
    end
end
