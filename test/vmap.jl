@testset "vmap" begin
    @testset for (f, x) in TestResources.UNARY_FUNCTIONS
        _, tape = trace(f, x; ctx=Taped.VMC())
        x_batch = Taped.Batch{Float64}(randn(3))
        vec_tape = Taped.vectorise(tape, f, x_batch)
        @test play!(vec_tape, f, x_batch).batch ≈ map(f, x_batch.batch)
    end
end
