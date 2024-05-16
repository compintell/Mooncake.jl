@testset "builtins" begin
    @test_throws(
        ErrorException,
        Tapir.rrule!!(CoDual(IntrinsicsWrappers.add_ptr, NoTangent()), 5.0, 4.0),
    )

    @test_throws(
        ErrorException,
        Tapir.rrule!!(CoDual(IntrinsicsWrappers.sub_ptr, NoTangent()), 5.0, 4.0),
    )

    @testset "_apply_iterate_equivalent with $(typeof(args))" for args in Any[
        (*, 5.0, 4.0),
        (*, (5.0, 4.0)),
        (*, [1.0, 2.0]),
        (*, 1.0, [2.0]),
        (*, [1.0, 2.0], ()),
    ]
        @test ==(
            Core._apply_iterate(Base.iterate, args...),
            Tapir._apply_iterate_equivalent(Base.iterate, args...),
        )
    end

    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:builtins))

    @testset "Disable casting to / from floats inside AD" begin
        @test_throws(
            ArgumentError,
            rrule!!(zero_fcodual(bitcast), zero_fcodual(Float64), zero_fcodual(5))
        )
        @test_throws(
            ArgumentError,
            rrule!!(
                zero_fcodual(Tapir.IntrinsicsWrappers.fptosi),
                zero_fcodual(Int),
                zero_fcodual(1.0),
            ),
        )
        @test_throws(
            ArgumentError,
            rrule!!(
                zero_fcodual(Tapir.IntrinsicsWrappers.fptoui),
                zero_fcodual(UInt),
                zero_fcodual(1.0),
            ),
        )
    end
end
