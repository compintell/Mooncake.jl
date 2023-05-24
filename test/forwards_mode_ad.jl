@testset "forwards_mode_ad" begin
    @testset for (f, x) in TestResources.UNARY_FUNCTIONS[1:4]
        _, tape = Umlaut.trace(f, x; ctx=Taped.FMC())
        fm_tape = to_forwards_mode_ad(tape, Dual(f, nothing), Dual(x, 1.0))
        y_dy = play!(fm_tape, Dual(f, nothing), Dual(x, 1.0))
        @test y_dy.x == f(x)
    end
end
