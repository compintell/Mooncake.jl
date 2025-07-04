using DispatchDoctor: @stable, @unstable, allow_unstable, TypeInstabilityError
using Test
using Mooncake
using StableRNGs: StableRNG

@stable stable_square(x) = x^2
@stable type_unstable_square(x) = x > 0 ? x^2 : 0
@stable default_union_limit = 2 type_unstable_square2(x) = x > 0 ? x^2 : 0

@testset "dispatch_doctor" begin
    @testset "stable function" begin
        f = stable_square
        x = 2.0

        result = f(x)
        @test result ≈ 4.0

        rule = Mooncake.build_rrule(Tuple{typeof(f),Float64})
        @test rule !== nothing
    end

    @testset "unstable function with allow_unstable" begin
        x = 2.0

        result = allow_unstable(() -> type_unstable_square(x))
        @test result ≈ 4.0

        result_neg = allow_unstable(() -> type_unstable_square(-1.0))
        @test result_neg ≈ 0.0

        allow_unstable() do
            rule = Mooncake.build_rrule(Tuple{typeof(type_unstable_square),Float64})
            @test rule !== nothing
        end
    end

    @testset "TypeInstabilityError without allow_unstable" begin
        x = 2.0

        @test_throws TypeInstabilityError type_unstable_square(x)

        rule = Mooncake.build_rrule(Tuple{typeof(type_unstable_square),Float64})
        @test rule !== nothing

        # However, note that type_unstable_square2 is _not_ marked unstable.
        # This should be preserved through the rrule.
        @test type_unstable_square2(x) ≈ 4.0
    end

    @testset "DispatchDoctor rrules exist" begin
        @stable simple_mult(x) = 3.0 * x
        @test simple_mult(2.0) ≈ 6.0

        rule = Mooncake.build_rrule(Tuple{typeof(simple_mult),Float64})
        @test rule !== nothing
    end

    @testset "derivatives are correct" begin
        rng = StableRNG(123)
        @testset "test_rule - $f with input $x" for x in [5.0, -5.0],
            f in [stable_square, type_unstable_square, type_unstable_square2]

            allow_unstable() do
                TestUtils.test_rule(rng, f, x; is_primitive=false)
            end
        end
    end
end
