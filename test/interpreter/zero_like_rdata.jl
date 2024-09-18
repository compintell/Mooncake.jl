@testset "zero_like_rdata" begin
    @testset "zero_like_rdata_from_type" begin
        @testset "$P" for P in Any[
            @NamedTuple{a},
            Tuple{Any},
            Float64,
            Int,
            Vector{Float64},
            TypeVar(:a),
        ]
            @test Mooncake.zero_like_rdata_from_type(P) isa Mooncake.zero_like_rdata_type(P)
        end
    end
end
