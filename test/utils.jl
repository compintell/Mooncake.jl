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
        @test map(sin, (5.0, 4.0)) == Tapir.tuple_map(sin, (5.0, 4.0))
        @test ==(
            map(*, (5, 4.0, 3), (5.0, 4, 3.0)),
            Tapir.tuple_map(*, (5, 4.0, 3), (5.0, 4, 3.0)),
        )

        @test map(sin, (a=5.0, b=4)) == Tapir.tuple_map(sin, (a=5.0, b=4))
        @test ==(
            map(*, (a=5, b=4.0, c=3), (a=5.0, b=4, c=3.0)),
            Tapir.tuple_map(*, (a=5, b=4.0, c=3), (a=5.0, b=4, c=3.0)),
        )

        # Require that length of arguments are equal.
        @test_throws ArgumentError Tapir.tuple_map(*, (5.0, 4.0), (4.0, ))
        @test_throws ArgumentError Tapir.tuple_map(*, (4.0, ), (5.0, 4.0))
    end
    @testset "_map_if_assigned!" begin
        @testset "unary bits type" begin
            x = Vector{Float64}(undef, 10)
            y = randn(10)
            z = Tapir._map_if_assigned!(sin, y, x)
            @test z === y
            @test all(map(isequal, z, map(sin, x)))
        end
        @testset "unary non bits type" begin
            x = Vector{Vector{Float64}}(undef, 2)
            x[1] = randn(5)
            y = [1.0, 1.0]
            z = Tapir._map_if_assigned!(sum, y, x)
            @test z === y

            # The first element of `x` is assigned, so z[1] should be its sum.
            @test z[1] ≈ sum(x[1])

            # The second element of `x` is unassigned, so z[2] should be unchanged.
            @test z[2] == 1.0
        end
        @testset "binary bits type" begin
            x1 = Vector{Float64}(undef, 7)
            x2 = Vector{Float64}(undef, 7)
            y = Vector{Float64}(undef, 7)
            z = Tapir._map_if_assigned!(*, y, x1, x2)
            @test z === y
            @test all(map(isequal, z, map(*, x1, x2)))
        end
        @testset "binary non bits type" begin
            x1 = Vector{Vector{Float64}}(undef, 2)
            x1[1] = randn(3)
            x2 = [randn(3), randn(2)]
            y = [1.0, 1.0]
            z = Tapir._map_if_assigned!(dot, y, x1, x2)
            @test z === y

            # The first element of x1 is assigned, so should have the inner product in z[1].
            @test z[1] ≈ dot(x1[1], x2[1])

            # The second element of x2 is not assigned, so z[2] should be unchanged.
            @test z[2] == 1
        end
    end
    @testset "_map" begin
        x = randn(10)
        y = randn(10)
        @test Tapir._map(*, x, y) == map(*, x, y)
        @assert length(map(*, x, randn(11))) == 10
        @test_throws AssertionError Tapir._map(*, x, randn(11)) 
    end
end
