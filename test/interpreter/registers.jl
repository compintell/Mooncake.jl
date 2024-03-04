@testset "registers" begin
    @test Taped.register_type(Float64) <: Taped.AugmentedRegister
    @test Taped.register_type(Bool) == Bool
    @test Taped.register_type(Any) == Any
    @test Taped.register_type(Real) == Any
    @test Taped.register_type(Union{Float64, Float32}) <: Taped.AugmentedRegister
    @test Taped.register_type(Union{Float64, Bool}) <: Union{Taped.AugmentedRegister, Bool}
end
