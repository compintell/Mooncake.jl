@testset "codual" begin
    @test CoDual(5.0, 4.0) isa CoDual{Float64,Float64}
    @test CoDual(Float64, NoTangent()) isa CoDual{Type{Float64},NoTangent}
    @test zero_codual(5.0) == CoDual(5.0, 0.0)

    @testset "$P" for (P, D) in Any[
        (Float64, CoDual{Float64,Float64}),
        (Int, CoDual{Int,NoTangent}),
        (Real, CoDual),
        (Any, CoDual),
        (Type{UnitRange{Int}}, CoDual{Type{UnitRange{Int}},NoTangent}),
        (Type{Tuple{T}} where {T}, CoDual),
        (Union{Float64,Int}, Union{CoDual{Float64,Float64},CoDual{Int,NoTangent}}),
        (UnionAll, CoDual),

        # Tuples:
        # Concrete tuples:
        (Tuple{Float64}, CoDual{Tuple{Float64},Tuple{Float64}}),
        (Tuple{Float64,Float32}, CoDual{Tuple{Float64,Float32},Tuple{Float64,Float32}}),
        (
            Tuple{Int,Float64,Float32},
            CoDual{Tuple{Int,Float64,Float32},Tuple{NoTangent,Float64,Float32}},
        ),

        # Small-Union Tuples
        (
            Tuple{Union{Float32,Float64}},
            Union{
                CoDual{Tuple{Float32},Tuple{Float32}},CoDual{Tuple{Float64},Tuple{Float64}}
            },
        ),
        (
            Tuple{Nothing,Union{Int,Float64}},
            Union{
                CoDual{Tuple{Nothing,Int},NoTangent},
                CoDual{Tuple{Nothing,Float64},Tuple{NoTangent,Float64}},
            },
        ),

        # General Abstract Tuples
        (Tuple{Any}, CoDual),
    ]
        @test codual_type(P) == D
    end

    @test Mooncake.fcodual_type(Type{Tuple{T}} where {T}) <: CoDual
    @testset "NoPullback" begin
        @test Base.issingletontype(typeof(NoPullback(zero_fcodual(5.0))))
        @test NoPullback(zero_codual(5.0))(4.0) == (0.0,)
    end
end
