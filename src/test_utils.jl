module TestUtils

using Taped, Umlaut
using Taped: CoDual, NoTangent

using Random, Test

has_equal_data(x::T, y::T) where {T<:String} = x == y
has_equal_data(x::Type, y::Type) = x == y
has_equal_data(x::Core.TypeName, y::Core.TypeName) = x == y
has_equal_data(x::Module, y::Module) = x == y
function has_equal_data(x::T, y::T) where {T<:Array}
    size(x) != size(y) && return false
    equality = map(1:length(x)) do n
        (isassigned(x, n) != isassigned(y, n)) && return false
        return (!isassigned(x, n) || has_equal_data(x[n], y[n]))
    end
    return all(equality)
end
function has_equal_data(x::T, y::T) where {T}
    isprimitivetype(T) && return isequal(x, y)
    return all(map(n -> has_equal_data(getfield(x, n), getfield(y, n)), fieldnames(T)))
end
has_equal_data(x::T, y::T) where {T<:Umlaut.Tape} = true

const AddressMap = Dict{Ptr{Nothing}, Ptr{Nothing}}

"""
    populate_address_map(primal, tangent)

Constructs an empty `AddressMap` and calls `populate_address_map!`.
"""
populate_address_map(primal, tangent) = populate_address_map!(AddressMap(), primal, tangent)

"""
    populate_address_map!(m::AddressMap, primal, tangent)

Fills `m` with pairs mapping from memory addresses in `primal` to corresponding memory
addresses in `tangent`. If the same memory address appears multiple times in `primal`,
throws an `AssertionError` if the same address is not mapped to in `tangent` each time.
"""
function populate_address_map!(m::AddressMap, primal::P, tangent::T) where {P, T}
    isprimitivetype(P) && return m
    if ismutabletype(P)
        @assert T <: MutableTangent
        k = pointer_from_objref(primal)
        v = pointer_from_objref(tangent)
        haskey(m, k) && (@assert (m[k] == v))
        m[k] = v
    end
    foreach(fieldnames(P)) do n
        populate_address_map!(m, getfield(primal, n), getfield(tangent.fields, n).tangent)
    end
    return m
end

function populate_address_map!(m::AddressMap, p::P, t) where {P<:Union{Tuple, NamedTuple}}
    foreach(n -> populate_address_map!(m, getfield(p, n), getfield(t, n)), fieldnames(P))
    return m
end

function populate_address_map!(m::AddressMap, p::Array, t::Array)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    populate_address_map!.(Ref(m), p, t)
    return m
end

populate_address_map!(m::AddressMap, p::Union{Core.TypeName, Type, Symbol, String}, t) = m

"""
    address_maps_are_consistent(x::AddressMap, y::AddressMap)

`true` if all keys in both `x` and `y` map to the same values. i.e. if any key appears in
both `x` and `y`, and the corresponding value is not the same in both `x` and `y`, return
`false`.
"""
function address_maps_are_consistent(x::AddressMap, y::AddressMap)
    return all(map(k -> x[k] == y[k], collect(intersect(keys(x), keys(y)))))
end

function test_rmad(rng::AbstractRNG, f, x...)

    # Run original function on deep-copies of inputs.
    x_correct = deepcopy(x)
    f_correct = f
    y_correct = f_correct(x_correct...)

    # Use finite differences to estimate vjps
    ẋ = randn_tangent(rng, x)
    ε = 1e-5
    x′ = _add_to_primal(x, _scale(ε, ẋ))
    y′ = f(x′...)
    ẏ = _scale(1 / ε, _diff(y′, y_correct))
    ẋ_post = _scale(1 / ε, _diff(x′, x_correct))

    # Run `rrule!!` on copies of `f` and `x`.
    f_f̄ = CoDual(f, zero_tangent(f))
    x_x̄ = map(x -> CoDual(deepcopy(x), zero_tangent(x)), x)
    inputs_address_map = populate_address_map(map(primal, x_x̄), map(shadow, x_x̄))
    y, pb!! = Taped.rrule!!(f_f̄, x_x̄...)

    # Query both `x_x̄` and `y`, because `x_x̄` may have been mutated by `f`.
    outputs_address_map = populate_address_map(
        (map(primal, x_x̄)..., primal(y)), (map(shadow, x_x̄)..., shadow(y)),
    )

    @test address_maps_are_consistent(inputs_address_map, outputs_address_map)

    # Verify that inputs / outputs are the same under `f` and its rrule.
    @test has_equal_data(x_correct, map(primal, x_x̄))
    @test has_equal_data(y_correct, primal(y))

    # Run reverse-pass.
    ȳ_delta = randn_tangent(rng, primal(y))
    x̄_delta = map(Base.Fix1(randn_tangent, rng), x)

    ȳ_init = set_to_zero!!(shadow(y))
    x̄_init = map(set_to_zero!! ∘ shadow, x_x̄)
    ȳ = increment!!(ȳ_init, ȳ_delta)
    x̄ = map(increment!!, x̄_init, x̄_delta)
    _, x̄... = pb!!(ȳ, shadow(f_f̄), x̄...)

    # Check that inputs have been returned to their original value.
    @test all(map(has_equal_data, x, map(primal, x_x̄)))

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

_deepcopy(x) = deepcopy(x)
_deepcopy(x::Module) = x

function test_rrule!!(
    rng::AbstractRNG, x...;
    interface_only=false,
    is_primitive=true,
    check_conditional_type_stability=true,
)
    # Set up problem.
    x_copy = (x[1], map(x -> _deepcopy(x isa CoDual ? primal(x) : x), x[2:end])...)
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
    x_p = (x_p[1], map(_deepcopy, x_p[2:end])...)
    check_stability = false
    if check_conditional_type_stability
        primal_is_type_stable = try
            is_inferred(apply, x_p...)
        catch e
            display(e)
            println()
            throw(ArgumentError("Primal evaluation does not work."))
        end

        # If the primal is type stable, and the user has requested that type-stability tests
        # be turned on, we check for type-stability when running the rrule!! and pullback.
        check_stability = check_conditional_type_stability && primal_is_type_stable
    end

    # Verify that the function to which the rrule applies is considered a primitive.
    is_primitive && @test Umlaut.isprimitive(Taped.RMC(), x_p...)

    # Run the rrule and extract results.
    y_ȳ, pb!! = check_stability ? (@inferred Taped.rrule!!(x_x̄...)) : Taped.rrule!!(x_x̄...)
    x = map(primal, x_x̄)
    x̄ = map(shadow, x_x̄)

    # Check output and incremented shadow types are correct.
    @test y_ȳ isa CoDual
    @test typeof(primal(y_ȳ)) == typeof(x_copy[1](map(_deepcopy, x_copy[2:end])...))
    if !interface_only
        @test has_equal_data(primal(y_ȳ), x_copy[1](map(_deepcopy, x_copy[2:end])...))
    end
    @test shadow(y_ȳ) isa tangent_type(typeof(primal(y_ȳ)))
    x̄_new = check_stability ? (@inferred pb!!(shadow(y_ȳ), x̄...)) : pb!!(shadow(y_ȳ), x̄...)
    @test all(map((a, b) -> typeof(a) == typeof(b), x̄_new, x̄))

    # Check aliasing.
    @test all(map((x̄, x̄_new) -> ismutable(x̄) ? x̄ === x̄_new : true, x̄, x̄_new))

    # Check that inputs have been returned to their original state.
    !interface_only && @test all(map(has_equal_data, x, x_copy))

    # Check that memory addresses have remained constant.
    new_x_addresses = map(get_address, x)
    @test all(map(==, x_addresses, new_x_addresses))

    # Check that the answers are numerically correct.
    !interface_only && test_rmad(rng, x...)
end

# Functionality for testing AD via Umlaut.
function test_taped_rrule!!(rng::AbstractRNG, f, x...; kwargs...)
    _, tape = trace(f, map(_deepcopy, x)...; ctx=Taped.RMC())
    f_t = Taped.UnrolledFunction(tape)
    test_rrule!!(
        rng, f_t, f, x...;
        is_primitive=false, check_conditional_type_stability=false, kwargs...,
    )
end

generate_args(::typeof(===), x) = [(x, 0.0), (1.0, x)]
function generate_args(::typeof(Core.ifelse), x)
    return [(true, x, 0.0), (false, x, 0.0), (true, 0.0, x), (false, 0.0, x)]
end
generate_args(::typeof(Core.sizeof), x) = [(x, )]
generate_args(::typeof(Core.svec), x) = [(x, ), (x, x)]
function generate_args(::typeof(getfield), x)
    names = fieldnames(typeof(x))
    return map(n -> (x, n), vcat(names..., eachindex(names)...))
end
generate_args(::typeof(isa), x) = [(x, Float64), (x, Int), (x, typeof(x))]
function generate_args(::typeof(setfield!), x)
    names = fieldnames(typeof(x))
    return map(n -> (x, n, getfield(x, n)), vcat(names..., eachindex(names)...))
end
generate_args(::typeof(tuple), x) = [(x, ), (x, x), (x, x, x)]
generate_args(::typeof(typeassert), x) = [(x, typeof(x))]
generate_args(::typeof(typeof), x) = [(x, )]

function functions_for_all_types()
    return [===, Core.ifelse, Core.sizeof, isa, tuple, typeassert, typeof]
end

functions_for_structs() = vcat(functions_for_all_types(), [getfield])

function functions_for_mutable_structs()
    return vcat(
        functions_for_structs(), [setfield!],# modifyfield!, replacefield!, swapfield!]
    )
end

"""
    test_rule_and_type_interactions(rng::AbstractRNG, x)

Check that a collection of standard functions for which we _ought_ to have a working rrule
for `x` work, and produce the correct answer. For example, the `rrule!!` for `typeof` should
work correctly on any type, we should have a working rule for `getfield` for any
struct-type, and we should have a rule for `setfield!` for any mutable struct type.

The purpose of this test is to ensure that, for any given `x`, the full range of primitive
functions that _ought_ to work on it, do indeed work on it.
"""
function test_rule_and_type_interactions(rng::AbstractRNG, x::P) where {P}

    # Generate standard test cases.
    fs = if ismutabletype(P)
        functions_for_mutable_structs()
    elseif isstructtype(P)
        functions_for_structs()
    else
        functions_for_all_types()
    end

    # Run standardised tests for all functions.
    @testset "$f" for f in fs
        arg_sets = generate_args(f, x)
        @testset for args in arg_sets
            test_rrule!!(
                rng, f, args...;
                interface_only=false,
                is_primitive=true,
                check_conditional_type_stability=false,
            )
        end
    end
end

end
