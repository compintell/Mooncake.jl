@testset "tracing" begin
    @testset for (interface_only, _r, f, x...) in TestResources.generate_test_functions()
        # val, tape = trace(f, x...; ctx=Taped.RMC())
        # @test val ≈ f(x...)
        # @test play!(tape, f, x...) ≈ f(x...)
        # compiled_tape = compile(tape)
        # @test compiled_tape(f, x...) ≈ f(x...)
    end
end
