@testset "interface" begin
    @testset "$(typeof((f, x...)))" for (ȳ, f, x...) in Any[
        (1.0, (x, y) -> x * y + sin(x) * cos(y), 5.0, 4.0),
        ([1.0, 1.0], x -> [sin(x), sin(2x)], 3.0),
    ]
        rule = build_rrule(f, x...)
        v, grad2 = value_and_pullback!!(rule, ȳ, f, x...)
        @test v ≈ f(x...)
    end
end
