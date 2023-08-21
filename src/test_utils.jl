module TestUtils

using Taped, Umlaut
using Taped: CoDual, NoTangent

using Random, Test

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
get_address(x) = ismutable(x) ? pointer_from_objref(x) : nothing

apply(f, x...) = f(x...)

function is_inferred(f::F, x...) where {F}
    static_type = only(Base.return_types(f, map(Core.Typeof, x)))
    dynamic_type = Core.Typeof(f(map(deepcopy, x)...))
    return static_type == dynamic_type
end

function conditional_stability(check_stability::Bool, f::F, x...) where {F}
    return check_stability ? (@inferred f(x...)) : f(x...)
end

function test_rrule!!(
    rng::AbstractRNG, x...;
    interface_only=false,
    is_primitive=true,
    check_conditional_type_stability=true,
)
    # Set up problem.
    x_copy = (x[1], map(x -> deepcopy(x isa CoDual ? primal(x) : x), x[2:end])...)
    x_addresses = map(get_address, x)
    x_x̄ = map(x -> x isa CoDual ? x : CoDual(x, randn_tangent(rng, x)), x)

    # Check that input types are valid.
    for x_x̄ in x_x̄
        @test typeof(shadow(x_x̄)) == tangent_type(typeof(primal(x_x̄)))
    end

    # Attempt to run primal programme. If the primal programme throws, display the original
    # exception and throw an additional exception which points this out.
    # Record whether or not it is type-stable.
    x_p = map(primal, x_x̄)
    x_p = (x_p[1], map(deepcopy, x_p[2:end])...)
    primal_is_type_stable = try
        is_inferred(apply, x_p...)
    catch e
        display(e)
        println()
        throw(ArgumentError("Primal evaluation does not work."))
    end

    # If the primal is type stable, and the user has requested that type-stability tests be
    # turned on, we check for type-stability when running the rrule!! and pullback.
    check_stability = check_conditional_type_stability && primal_is_type_stable

    # Verify that the function to which the rrule applies is considered a primitive.
    is_primitive && @test Umlaut.isprimitive(Taped.RMC(), x_p...)

    # Run the rrule and extract results.
    y_ȳ, pb!! = check_stability ? (@inferred Taped.rrule!!(x_x̄...)) : Taped.rrule!!(x_x̄...)
    x = map(primal, x_x̄)
    x̄ = map(shadow, x_x̄)

    # Check output and incremented shadow types are correct.
    @test y_ȳ isa CoDual
    @test typeof(primal(y_ȳ)) == typeof(x_copy[1](map(deepcopy, x_copy[2:end])...))
    !interface_only && @test primal(y_ȳ) == x_copy[1](map(deepcopy, x_copy[2:end])...)
    @test shadow(y_ȳ) isa tangent_type(typeof(primal(y_ȳ)))
    x̄_new = check_stability ? (@inferred pb!!(shadow(y_ȳ), x̄...)) : pb!!(shadow(y_ȳ), x̄...)
    @test all(map((a, b) -> typeof(a) == typeof(b), x̄_new, x̄))

    # Check aliasing.
    @test all(map((x̄, x̄_new) -> ismutable(x̄) ? x̄ === x̄_new : true, x̄, x̄_new))

    # Check that inputs have been returned to their original state.
    !interface_only && @test all(map(==, x, x_copy))

    # Check that memory addresses have remained constant.
    new_x_addresses = map(get_address, x)
    @test all(map(==, x_addresses, new_x_addresses))

    # Check that the answers are numerically correct.
    !interface_only && test_rmad(rng, x...)
end

# Functionality for testing AD via Umlaut.
function test_taped_rrule!!(rng::AbstractRNG, f, x...; kwargs...)
    _, tape = trace(f, map(deepcopy, x)...; ctx=Taped.RMC())
    f_t = Taped.UnrolledFunction(tape)
    test_rrule!!(rng, f_t, f, x...; is_primitive=false, kwargs...)
end

end
