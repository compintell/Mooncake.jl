@testset "tracing" begin
    @testset for (interface_only, f, x...) in TestResources.TEST_FUNCTIONS
        val, tape = trace(f, x...; ctx=Taped.TC())
        @test val ≈ f(x...)
        @test play!(tape, f, x...) ≈ f(x...)
        compiled_tape = compile(tape)
        @test compiled_tape(f, x...) ≈ f(x...)
    end
end
