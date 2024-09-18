@testset "builtins" begin
    @test_throws(
        ErrorException,
        Mooncake.rrule!!(CoDual(IntrinsicsWrappers.add_ptr, NoTangent()), 5.0, 4.0),
    )

    @test_throws(
        ErrorException,
        Mooncake.rrule!!(CoDual(IntrinsicsWrappers.sub_ptr, NoTangent()), 5.0, 4.0),
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
            Mooncake._apply_iterate_equivalent(Base.iterate, args...),
        )
    end

    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:builtins))

    # Unhandled built-in throws an intelligible error.
    @test_throws(
        Mooncake.MissingRuleForBuiltinException,
        invoke(Mooncake.rrule!!, Tuple{CoDual{<:Core.Builtin}}, zero_fcodual(getfield)),
    )

    # Unhandled intrinsic throws an intelligible error.
    @test_throws(
        Mooncake.IntrinsicsWrappers.MissingIntrinsicWrapperException,
        invoke(Mooncake.IntrinsicsWrappers.translate, Tuple{Any}, Val(:foo)),
    )

    @testset "Disable bitcast to differentiable type" begin
        @test_throws(
            ArgumentError,
            rrule!!(zero_fcodual(bitcast), zero_fcodual(Float64), zero_fcodual(5))
        )
    end
end
