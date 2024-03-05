@testset "registers" begin
    @test Taped.register_type(Float64) <: Taped.AugmentedRegister{CoDual{Float64, Float64}}
    @test Taped.register_type(Bool) <: Taped.AugmentedRegister{CoDual{Bool, NoTangent}}
    @test Taped.register_type(Any) == Taped.AugmentedRegister
    @test Taped.register_type(Real) == Taped.AugmentedRegister
    @test Taped.register_type(Union{Float64, Float32}) <: Taped.AugmentedRegister
    @test Taped.register_type(Union{Float64, Bool}) <: Union{Taped.AugmentedRegister, Bool}
end
