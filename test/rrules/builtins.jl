foo_throws(e) = throw(e)

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

    @testset "is_homogeneous_and_immutable" begin
        x = Tuple(randn(1000))
        @test @inferred Mooncake.is_homogeneous_and_immutable(x)
        @test (@allocations Mooncake.is_homogeneous_and_immutable(x)) == 0
    end

    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:builtins))

    # Unhandled built-in throws an intelligible error.
    @test_throws(
        Mooncake.MissingRuleForBuiltinException,
        invoke(Mooncake.rrule!!, Tuple{CoDual{<:Core.Builtin}}, zero_fcodual(getfield)),
    )

    # Check that Base.showerror runs.
    @test ==(
        showerror(IOBuffer(; write=true), Mooncake.MissingRuleForBuiltinException("hmm")),
        nothing,
    )

    # Unhandled intrinsic throws an intelligible error.
    @test_throws(
        Mooncake.IntrinsicsWrappers.MissingIntrinsicWrapperException,
        invoke(Mooncake.IntrinsicsWrappers.translate, Tuple{Any}, Val(:foo)),
    )

    @testset "Disable bitcast to differentiable type & Int->Ptr" begin
        @test_throws(
            ArgumentError,
            rrule!!(zero_fcodual(bitcast), zero_fcodual(Float64), zero_fcodual(5))
        )
        @test_throws(
            ArgumentError,
            rrule!!(zero_fcodual(bitcast), zero_fcodual(Ptr{Float64}), zero_fcodual(5))
        )
    end

    @testset "bitcast for Ptr->Ptr" begin
        test_cases = [(
            zero_fcodual(bitcast),
            zero_fcodual(Ptr{Float64}),
            CoDual(Ptr{Float32}(5), Ptr{Float32}(5)),
        )]

        map(test_cases) do (Intrinsic_bitcast, bitpattern_type, val)
            res, pb = rrule!!(Intrinsic_bitcast, bitpattern_type, val)
            @test pb isa Mooncake.NoPullback
            @test res == CoDual(Ptr{Float64}(5), Ptr{Float64}(5))
        end
    end

    @testset "throw" begin
        # Throw primitive continues to throw the exception it is meant to.
        @test_throws(
            ArgumentError,
            Mooncake.rrule!!(zero_fcodual(throw), zero_fcodual(ArgumentError("hello")))
        )
        @test_throws(
            AssertionError,
            Mooncake.rrule!!(zero_fcodual(throw), zero_fcodual(AssertionError("hello")))
        )

        # Derived rule throws the correct exception.
        rule_arg = Mooncake.build_rrule(Tuple{typeof(foo_throws),ArgumentError})
        @test_throws(
            ArgumentError,
            rule_arg(zero_fcodual(foo_throws), zero_fcodual(ArgumentError("hello")))
        )
        rule_assert = Mooncake.build_rrule(Tuple{typeof(foo_throws),AssertionError})
        @test_throws(
            AssertionError,
            rule_assert(zero_fcodual(foo_throws), zero_fcodual(AssertionError("hmmm")))
        )
    end
end
