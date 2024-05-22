module FwdsRvsDataTestResources
    struct Foo{A} end
end

@testset "fwds_rvs_data" begin
    @testset "$(typeof(p))" for (_, p, _...) in Tapir.tangent_test_cases()
        TestUtils.test_fwds_rvs_data(Xoshiro(123456), p)
    end
    @testset "zero_rdata_from_type checks" begin
        @test Tapir.can_produce_zero_rdata_from_type(Vector) == true
        @test Tapir.zero_rdata_from_type(Vector) == NoRData()
        @test Tapir.can_produce_zero_rdata_from_type(FwdsRvsDataTestResources.Foo) == false
        @test Tapir.can_produce_zero_rdata_from_type(Tuple{Float64, Type{Float64}})
        @test ==(
            Tapir.zero_rdata_from_type(FwdsRvsDataTestResources.Foo),
            Tapir.CannotProduceZeroRDataFromType(),
        )
        @test !Tapir.can_produce_zero_rdata_from_type(Tuple)
    end
    @testset "lazy construction checks" begin
        # Check that lazy construction is in fact lazy for some cases where performance
        # really matters -- floats, things with no rdata, etc.
        @testset "$p" for (p, fully_lazy) in Any[
            (5, true),
            (Int32(5), true),
            (5.0, true),
            (5f0, true),
            (Float16(5.0), true),
            (StructFoo(5.0), false),
            (StructFoo(5.0, randn(4)), false),
            (Bool, true),
        ]
            @test fully_lazy == Base.issingletontype(typeof(LazyZeroRData(p)))
            @inferred Tapir.instantiate(LazyZeroRData(p))
        end
    end
end
