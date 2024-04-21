@testset "zero_like_rdata" begin
    @testset "zero_like_rdata_from_type" begin
        @testset "$P" for P in Any[
            @NamedTuple{a},
            Tuple{Any},
        ]
            @test Tapir.zero_like_rdata_from_type(P) isa Tapir.ZeroRData
        end
    end
end
