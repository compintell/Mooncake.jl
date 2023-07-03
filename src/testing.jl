function test_rmad(rng::AbstractRNG, f, x...)

    # Run original function on deep-copies of inputs.
    x_correct = deepcopy(x)
    f_correct = deepcopy(f)
    y_correct = f_correct(x_correct...)

    # Run `rrule!!` on copies of `f` and `x`.
    f_f̄ = CoDual(deepcopy(f), zero_tangent(f))
    x_x̄ = map(x -> CoDual(deepcopy(x), zero_tangent(x)), x)
    x̄_init = map(deepcopy ∘ shadow, x_x̄)
    y, pb!! = rrule!!(f_f̄, x_x̄...)

    # Verify that inputs / outputs are the same under `f` and its rrule.
    @test x_correct == map(primal, x_x̄)
    @test y_correct == primal(y)

    # Run reverse-pass.
    ȳ = increment!!(shadow(y), randn_tangent(rng, primal(y)))
    ȳ_init = deepcopy(ȳ)
    _, x̄... = pb!!(ȳ, shadow(f_f̄), map(shadow, x_x̄)...)

    # Check that inputs have been returned to their original value.
    @test all(map(isequal, x, map(primal, x_x̄)))

    # Use finite differences to estimate vjps
    ẋ = randn_tangent(rng, x)
    ε = 1e-3
    x′ = _add_to_primal(x, _scale(ε, ẋ))
    y′ = f(x′...)
    ẏ = _scale(1 / ε, _diff(y′, y_correct))

    # pullbacks increment, so have to compare to the incremented quantity.
    @test _dot(ȳ_init, ẏ) + _dot(x̄_init, ẋ) ≈ _dot(x̄, ẋ) rtol=1e-3 atol=1e-3
end
