@testset "registers" begin
    @test Phi.register_type(Float64) <: Phi.AugmentedRegister{CoDual{Float64, Float64}}
    @test Phi.register_type(Bool) <: Phi.AugmentedRegister{CoDual{Bool, NoTangent}}
    @test Phi.register_type(Any) == Phi.AugmentedRegister
    @test Phi.register_type(Real) == Phi.AugmentedRegister
    @test ==(Phi.register_type(Union{Float64, Float32}), Phi.AugmentedRegister)
    @test Phi.register_type(Union{Float64, Bool}) <: Union{Phi.AugmentedRegister, Bool}
end
