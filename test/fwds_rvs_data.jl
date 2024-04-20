@testset "fwds_rvs_data" begin
    @testset "$(typeof(p))" for (_, p, _...) in Tapir.tangent_test_cases()
        TestUtils.test_fwds_rvs_data(Xoshiro(123456), p)
    end
    @testset "rdata_interal_ctors" begin
        @testset "$p" for (p, fully_lazy) in Any[
            (5, true),
            (Int32(5), true),
            (5.0, true),
            (5f0, true),
            (Float16(5.0), true),
            (StructFoo(5.0), false),
            (StructFoo(5.0, randn(4)), false),
        ]
            r = LazyZeroRData(p)
            @test zero_rdata(p) == instantiate(r)
            @test fully_lazy == Base.issingletontype(typeof(r))
        end
    end
end
