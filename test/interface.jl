@testset "interface" begin
    @testset "$(typeof((f, x...)))" for (ȳ, f, x...) in Any[
        (1.0, (x, y) -> x * y + sin(x) * cos(y), 5.0, 4.0),
        ([1.0, 1.0], x -> [sin(x), sin(2x)], 3.0),
        (1.0, x -> sum(5x), [5.0, 2.0]),
    ]
        rule = build_rrule(f, x...)
        v, (df, dx...) = value_and_pullback!!(rule, ȳ, f, x...)
        @test v ≈ f(x...)
        @test df isa tangent_type(typeof(f))
        for (_dx, _x) in zip(dx, x)
            @test _dx isa tangent_type(typeof(_x))
        end
    end
end
