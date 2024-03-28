@testset "registers" begin
    @test Tapir.register_type(Float64) <: Tapir.AugmentedRegister{CoDual{Float64, Float64}}
    @test Tapir.register_type(Bool) <: Tapir.AugmentedRegister{CoDual{Bool, NoTangent}}
    @test Tapir.register_type(Any) == Tapir.AugmentedRegister
    @test Tapir.register_type(Real) == Tapir.AugmentedRegister
    @test ==(Tapir.register_type(Union{Float64, Float32}), Tapir.AugmentedRegister)
    @test Tapir.register_type(Union{Float64, Bool}) <: Union{Tapir.AugmentedRegister, Bool}
end
