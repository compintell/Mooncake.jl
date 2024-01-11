@testset "ir_normalisation" begin
    @testset "ensure_single_argument_usage_per_call!" begin
        ir = Base.code_ircode(x -> x * x, Tuple{Float64})
        ir = CC.IRCode()

    end
    @testset "invokes_to_calls!" begin

        # Generate ir with an invoke in it, and verify that it runs.
        mi = which(Tuple{typeof(sin), Float64}).specializations
        ir = Taped.ircode(
            Any[
                Expr(:invoke, mi, sin, 5.0),
                ReturnNode(SSAValue(1)),
            ],
            Any[Tuple{}],
        )
        @assert Core.OpaqueClosure(ir)() == sin(5.0)

        # Transform invokes to calls and verify the results.
        Taped.invokes_to_calls!(ir)
        @test ir.stmts.inst[1] == Expr(:call, sin, 5.0)
        @test Core.OpaqueClosure(ir)() == sin(5.0)
    end
    @testset "foreigncall_exprs_to_call_exprs!" begin
        expr = Expr(
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
        Taped.__foreigncall_expr_to_call_expr!(expr, sp_map)
        @test Meta.isexpr(expr, :call)
        @test expr.args[1] == Taped._foreigncall_
    end
    @testset "new_exprs_to_call_exprs!" begin
        args = Any[GlobalRef(Taped, :Foo), SSAValue(1), :hi]
        ex = Expr(:new, args...)
        Taped.__new_expr_to_call_expr!(ex)
        @test Meta.isexpr(ex, :call)
        @test ex.args[1] == Taped._new_
        @test ex.args[2:end] == args
    end
end
