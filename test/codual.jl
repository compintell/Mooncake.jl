@testset "reverse_mode_ad" begin
    @test CoDual(5.0, 4.0) isa CoDual{Float64, Float64}
    @test CoDual(Float64, NoTangent()) isa CoDual{Type{Float64}, NoTangent}
    @test zero_codual(5.0) == CoDual(5.0, 0.0)
    @test Phi.uninit_codual(5.0) == CoDual(5.0, 0.0)
    @test codual_type(Float64) == CoDual{Float64, Float64}
    @test codual_type(Int) == CoDual{Int, NoTangent}
    @test codual_type(Real) == CoDual
    @test codual_type(Any) == CoDual
    @test codual_type(Type{UnitRange{Int}}) == CoDual{Type{UnitRange{Int}}, NoTangent}
    @test(==(
        codual_type(Union{Float64, Int}),
        Union{CoDual{Float64, Float64}, CoDual{Int, NoTangent}},
    ))
end
