@testset "algorithmic_differentiation" begin
    @testset for (f, x) in TestResources.UNARY_FUNCTIONS
        val, tape = Taped.taped_trace(f, x; ctx=Taped.RMADContext())
        @test val ≈ f(x)
        @test play!(tape, f, x) ≈ f(x)
        compiled_tape = compile(tape)
        @test compiled_tape(f, x) ≈ f(x)


        # You'll only be able to run ReverseDiff on the compiled tape if you modify the
        # implementation of `Umlaut.compile` to not restrict the type of the compiled
        # function to only the arguments on which it was originally called.
        @test_broken isapprox(
            ReverseDiff.gradient(x -> compiled_tape(f, only(x)), [x]),
            ReverseDiff.gradient(x -> f(only(x)), [x]),
        )
    end

    @testset "value-dependent control flow" begin
        f = TestResources.value_dependent_control_flow
        x = 5.0
        ctx = Taped.RMADContext()
        val, tape = Taped.taped_trace(f, x, 5; ctx)
        @test_throws ErrorException play!(tape, f, x, 4)
    end

    @testset "value_and_derivative($f, $x)" for (f, x) in TestResources.UNARY_FUNCTIONS
        @test isapprox(
            only(ReverseDiff.gradient(x -> f(only(x)), [x])),
            last(Taped.value_and_derivative(f, x)),
        )
    end
end
