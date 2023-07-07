function test_rmad(rng::AbstractRNG, f, x...)

    # Run original function on deep-copies of inputs.
    x_correct = deepcopy(x)
    f_correct = f
    y_correct = f_correct(x_correct...)

    # Use finite differences to estimate vjps
    ẋ = randn_tangent(rng, x)
    # ẋ = zero_tangent(x)
    ε = 1e-5
    x′ = _add_to_primal(x, _scale(ε, ẋ))
    y′ = f(x′...)
    ẏ = _scale(1 / ε, _diff(y′, y_correct))
    ẋ_post = _scale(1 / ε, _diff(x′, x_correct))

    # Run `rrule!!` on copies of `f` and `x`.
    f_f̄ = CoDual(f, zero_tangent(f))
    x_x̄ = map(x -> CoDual(deepcopy(x), zero_tangent(x)), x)
    y, pb!! = Taped.rrule!!(f_f̄, x_x̄...)

    # Verify that inputs / outputs are the same under `f` and its rrule.
    @test x_correct == map(primal, x_x̄)
    @test y_correct == primal(y)

    # Run reverse-pass.
    ȳ_delta = randn_tangent(rng, primal(y))
    x̄_delta = map(Base.Fix1(randn_tangent, rng), x)
    # ȳ_delta = zero_tangent(primal(y))
    # x̄_delta = map(zero_tangent, x)

    ȳ_init = set_to_zero!!(shadow(y))
    x̄_init = map(set_to_zero!! ∘ shadow, x_x̄)
    ȳ = increment!!(ȳ_init, ȳ_delta)
    x̄ = map(increment!!, x̄_init, x̄_delta)
    _, x̄... = pb!!(ȳ, shadow(f_f̄), x̄...)

    # Check that inputs have been returned to their original value.
    @test all(map(isequal, x, map(primal, x_x̄)))

    # pullbacks increment, so have to compare to the incremented quantity.
    # @show ȳ_delta
    # @show x̄_delta
    # @show x̄
    @test _dot(ȳ_delta, ẏ) + _dot(x̄_delta, ẋ_post) ≈ _dot(x̄, ẋ) rtol=1e-3 atol=1e-3
end

test_alias(x::Vector{Float64}) = x

function rrule!!(::CoDual{typeof(test_alias)}, x::CoDual)
    function test_alias_pullback!!(ȳ::Vector{Float64}, ::NoTangent, x̄::Vector{Float64})
        @assert ȳ === x̄
        return NoTangent(), ȳ
    end
    return x, test_alias_pullback!!
end
