@testset "forwards_mode_ad" begin
    @testset for (f, x) in TestResources.UNARY_FUNCTIONS[1:2]
        _, tape = Umlaut.trace(f, x; ctx=Taped.FMC())
        fm_tape = to_forwards_mode_ad(tape)
        display(fm_tape)
        println()
        y_dy = play!(fm_tape, Dual(f, nothing), Dual(x, 1.0))
        @test y_dy.x == f(x)
        @test y_dy.dx ≈ only(ReverseDiff.gradient(x -> f(only(x)), [x]))
    end
end
