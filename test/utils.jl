@testset "utils" begin
    @testset "_typeof" begin
        @test _typeof(5.0) == Float64
        @test _typeof(randn(1)) == Vector{Float64}
        @test _typeof(Float64) == Type{Float64}
        @test _typeof(Vector{Int}) == Type{Vector{Int}}
        @test _typeof(Vector{T} where {T}) == Type{Vector}
        @test _typeof((5.0, Float64)) == Tuple{Float64, Type{Float64}}
        @test _typeof((a=5.0, b=Float64)) == @NamedTuple{a::Float64, b::Type{Float64}}
    end
    @testset "tuple_map" begin
        @test map(sin, (5.0, 4.0)) == Phi.tuple_map(sin, (5.0, 4.0))
        @test ==(
            map(*, (5, 4.0, 3), (5.0, 4, 3.0)),
            Phi.tuple_map(*, (5, 4.0, 3), (5.0, 4, 3.0)),
        )
    end
end
