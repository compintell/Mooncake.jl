@testset "tracing" begin
    @testset for (f, x) in TestResources.UNARY_FUNCTIONS[1:1]
        val, tape = trace(f, x; ctx=Taped.TapedContext())
        @test val ≈ f(x)
        @test play!(tape, f, x) ≈ f(x)
        compiled_tape = compile(tape)
        @test compiled_tape(f, x) ≈ f(x)
    end
end
