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
    @testset "misc fdata / rdata type checking" begin
        @test(==(
            Tapir.rdata_type(tangent_type(Tuple{Union{Float32, Float64}})),
            Tuple{Union{Float32, Float64}},
        ))
        @test(==(
            Tapir.rdata_type(tangent_type(Tuple{Union{Int32, Int}})), NoRData
        ))
        @test(==(
            Tapir.rdata_type(tangent_type(
                Tuple{Union{Vector{Float32}, Vector{Float64}}}
            )),
            NoRData,
        ))
    end

    # Tests that the static type of an fdata / rdata is correct happen in
    # test_fwds_rvs_data, so here we only need to test the specific quirks for a given type.
    @testset "fdata and rdata verification" begin
        @testset "Array" begin
            @test_throws InvalidFDataException verify_fdata_value(randn(10), randn(11))
            @test_throws InvalidFDataException verify_fdata_value([randn(10)], [randn(11)])
            @test_throws InvalidFDataException verify_fdata_value(Any[1], [NoFData()])
        end
        @testset "Tuple" begin
            @test_throws InvalidFDataException verify_fdata_value((), ())
            @test_throws InvalidFDataException verify_fdata_value((5,), (NoFData(), ))
            @test_throws InvalidRDataException verify_rdata_value((), ())
            @test_throws InvalidRDataException verify_rdata_value((5,), (NoRData(), ))
        end
        @testset "Ptr" begin
            @test verify_fdata_value(Ptr{Float64}(), Ptr{Float64}()) === nothing
            @test verify_rdata_value(Ptr{Float64}(), NoRData()) === nothing
        end
    end
end
