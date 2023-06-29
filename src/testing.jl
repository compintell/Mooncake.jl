function test_rmad(rng::AbstractRNG, f, x::Union{Real, Array{<:Real}}...)
    y_correct = f(deepcopy(x)...)
    x_copy = deepcopy(x)
    y, pb!! = rrule!!(f, x_copy...)
    ȳ = randn_tangent(rng, y)
    f̄, x̄s... = pb!!(ȳ)

    @test y_correct ≈ y
    @test all(map(isapprox, x, x̄s))
end
