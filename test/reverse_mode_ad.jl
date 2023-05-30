@testset "reverse_mode_ad" begin
    @testset for (f, x) in TestResources.UNARY_FUNCTIONS[1:2]
        _, tape = Umlaut.trace(f, x; ctx=Taped.RMC())
        x_dx = Shadow(x, Ref(0.0), nothing)
        rm_tape = to_reverse_mode_ad(tape)
        play!(rm_tape, f, x_dx)
        @test ReverseDiff.gradient(x -> f(only(x)), [x])[1] ≈ Taped.shadow(x_dx)[]
    end
end
