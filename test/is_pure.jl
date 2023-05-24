@testset "is_pure" begin
    @test Taped.is_pure(sin, 5.0)
    @test Taped.is_pure(cos, 5f0)

    _, tape = Taped.trace(TestResources.test_one, 5.0; ctx=Taped.VMC())
    @test Taped.is_pure(tape)
end
