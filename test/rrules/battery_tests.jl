@testset "battery_tests" begin
    @testset "$(typeof(x))" for x in vcat(
        [
            true, false,
            UInt8(0), UInt8(3),
            UInt16(0), UInt16(5),
            UInt32(0), UInt32(7),
            UInt64(0), UInt64(9),
            UInt128(0), UInt128(3),
            Int8(0), Int8(3),
            Int16(0), Int16(-1),
            Int32(0), Int32(-3),
            Int64(0), Int64(5),
            Int128(0), Int128(24),
            "hello",
        ],
        randn(Float64, 5),
        # randn(Float32, 5),
        # randn(Float16, 5),
        [
            Adjoint(randn(2, 2)),
            Diagonal(randn(2)),
            UnitRange(1, 3),
            Transpose(randn(2, 2)),
            view(randn(3, 3), 1:2, 1:1),
            Xoshiro(123456),
            Ref(5.0),
            TestResources.StructFoo(5.0, randn(5)),
            TestResources.MutableFoo(5.0, randn(5)),
        ]
    )
        TestUtils.test_rule_and_type_interactions(Xoshiro(123456), x)
    end
end
