@testset "interface" begin
    @testset "$(typeof((f, x...)))" for (ȳ, f, x...) in Any[
        (1.0, (x, y) -> x * y + sin(x) * cos(y), 5.0, 4.0),
        ([1.0, 1.0], x -> [sin(x), sin(2x)], 3.0),
        (1.0, x -> sum(5x), [5.0, 2.0]),
    ]
        @testset "debug_mode=$debug_mode" for debug_mode in Bool[false, true]
            rule = build_rrule(f, x...; debug_mode)
            v, (df, dx...) = value_and_pullback!!(rule, ȳ, f, x...)
            @test v ≈ f(x...)
            @test df isa tangent_type(typeof(f))
            for (_dx, _x) in zip(dx, x)
                @test _dx isa tangent_type(typeof(_x))
            end
        end
    end
    @testset "sensible error when CoDuals are passed to `value_and_pullback!!" begin
        foo(x) = sin(cos(x))
        rule = build_rrule(foo, 5.0)
        @test_throws ArgumentError value_and_pullback!!(rule, 1.0, foo, CoDual(5.0, 0.0))
    end
    @testset "value_and_gradient!!" begin
        @testset "($(typeof(fargs))" for fargs in Any[
            (sin, randn(Float64)),
            (sin, randn(Float32)),
            (x -> sin(cos(x)), randn(Float64)),
            (x -> sin(cos(x)), randn(Float32)),
            ((x, y) -> x + sin(y), randn(Float64), randn(Float64)),
            ((x, y) -> x + sin(y), randn(Float32), randn(Float32)),
            ((x...) -> x[1] + x[2], randn(Float64), randn(Float64)),
        ]
            rule = build_rrule(fargs...)
            f, args... = fargs
            v, dfargs = value_and_gradient!!(rule, fargs...)
            @test v == f(args...)
            for (arg, darg) in zip(fargs, dfargs)
                @test tangent_type(typeof(arg)) == typeof(darg)
            end
        end

        rule = build_rrule(identity, (5.0, 4.0))
        @test_throws(
            Mooncake.ValueAndGradientReturnTypeError,
            value_and_gradient!!(rule, identity, (5.0, 4.0)),
        )
    end
end
