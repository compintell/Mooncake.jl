@testset "interface" begin
    f = (x, y) -> x * y + sin(x) * cos(y)
    x = 5.0
    y = 4.0
    rule = build_rrule(f, x, y)
    v, grad = value_and_gradient!!(rule, f, x, y)
    @test v ≈ f(x, y)
    @test grad isa Tuple{NoTangent, Float64, Float64}

    v, grad2 = value_and_pullback!!(rule, 1.0, f, x, y)
    @test v ≈ f(x, y)
    @test grad == grad2
end
