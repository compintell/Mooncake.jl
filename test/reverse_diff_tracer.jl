@testset "reverse_diff_tracer" begin
    @testset for (f, x) in TestResources.UNARY_FUNCTIONS[1:4]
        _, tape = Umlaut.trace(f, x; ctx=Taped.RDTC())
        rd_via_tape = only(ReverseDiff.gradient(x -> play!(tape, f, only(x)), [x]))
        rd = only(ReverseDiff.gradient(x -> f(only(x)), [x]))
        @test rd_via_tape ≈ rd
    end
    @testset "value-dependent control flow" begin
        n = Ref(5)
        g_tape = ReverseDiff.GradientTape(
            x -> TestResources.value_dependent_control_flow(only(x), n[]), [5.0]
        )
        fast_tape = ReverseDiff.compile(g_tape)
        @show ReverseDiff.gradient!(fast_tape, [5.0])
        n[] = 4
        @show ReverseDiff.gradient!(fast_tape, [5.0])
        @show ReverseDiff.gradient(x -> TestResources.value_dependent_control_flow(only(x), n[]), [5.0])
        # How can I keep the items on the tape?
    end
end
