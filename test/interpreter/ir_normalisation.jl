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
        ir = Taped.__invokes_to_calls!(ir)
        @test ir.stmts.inst[1] == Expr(:call, sin, 5.0)
        @test Core.OpaqueClosure(ir)() == sin(5.0)
    end
end
