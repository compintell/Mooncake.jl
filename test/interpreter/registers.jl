@testset "registers" begin
    @test Taped.register_type(Float64) <: Taped.AugmentedRegister
    @test Taped.register_type(Bool) == Bool
end
