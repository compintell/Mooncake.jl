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
    return all(map(
        n -> isdefined(x, n) ? has_equal_data(getfield(x, n), getfield(y, n)) : true,
        fieldnames(T),
    ))
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
    foreach(n -> isassigned(p, n) && populate_address_map!(m, p[n], t[n]), eachindex(p))
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
    x̄_delta = map(Base.Fix1(randn_tangent, rng) ∘ primal, x_x̄)

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

    # Run the primal computation, and compute the expected tangent type.
    y_primal = x_copy[1](map(_deepcopy, x_copy[2:end])...)
    Ty = Core.Typeof(y_primal)
    Tȳ = tangent_type(Ty)

    # Run the rrule and extract results.
    rrule_ret = check_stability ? (@inferred Taped.rrule!!(x_x̄...)) : Taped.rrule!!(x_x̄...)
    @test rrule_ret isa Tuple{CoDual{Ty, Tȳ}, Any}
    y_ȳ, pb!! = rrule_ret
    x = map(primal, x_x̄)
    x̄ = map(shadow, x_x̄)

    # Check output and incremented shadow types are correct.
    @test y_ȳ isa CoDual
    @test typeof(primal(y_ȳ)) == typeof(y_primal)
    if !interface_only
        @test has_equal_data(primal(y_ȳ), y_primal)
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
function test_taped_rrule!!(rng::AbstractRNG, f, x...; interface_only=false, kwargs...)

    # Try to run the primal, just to make sure that we're not calling it on bad inputs.
    f(_deepcopy(x)...)

    _, tape = trace(f, map(_deepcopy, x)...; ctx=Taped.RMC())
    f_t = Taped.UnrolledFunction(tape)

    # Check that the gradient is self-consistent.
    test_rrule!!(
        rng, f_t, f, x...;
        is_primitive=false,
        check_conditional_type_stability=false,
        interface_only,
        kwargs...,
    )

    # Check that f_t remains a faithful representation of the original function.
    if !interface_only
        @test has_equal_data(f(deepcopy(x)...), play!(f_t.tape, f, deepcopy(x)...))
    end
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

function test_tangent(rng::AbstractRNG, p::P, z_target::T, x::T, y::T) where {P, T}

    # Verify that interface `tangent_type` runs.
    Tt = tangent_type(P)
    t = randn_tangent(rng, p)
    z = zero_tangent(p)

    # Check that user-provided tangents have the same type as `tangent_type` expects.
    @test T == Tt

    # Check that ismutabletype(P) => ismutabletype(T)
    if ismutabletype(P) && !(Tt == NoTangent)
        @test ismutabletype(Tt)
    end

    # Check that tangents are of the correct type.
    @test Tt == typeof(t)
    @test Tt == typeof(z)

    # Check that zero_tangent is deterministic.
    @test has_equal_data(z, Taped.zero_tangent(p))

    # Check that zero_tangent infers.
    @test has_equal_data(z, @inferred Taped.zero_tangent(p))

    # Verify that the zero tangent is zero via its action.
    zc = deepcopy(z)
    tc = deepcopy(t)
    @test has_equal_data(@inferred(increment!!(zc, zc)), zc)
    @test has_equal_data(increment!!(zc, tc), tc)
    @test has_equal_data(increment!!(tc, zc), tc)

    if ismutabletype(P)
        @test increment!!(zc, zc) === zc
        @test increment!!(tc, zc) === tc
        @test increment!!(zc, tc) === zc
        @test increment!!(tc, tc) === tc
    end

    z_pred = increment!!(x, y)
    @test has_equal_data(z_pred, z_target)
    if ismutabletype(P)
        @test z_pred === x
    end

    # If t isn't the zero element, then adding it to itself must change its value.
    if t != z
        if !ismutabletype(P)
            @test !has_equal_data(increment!!(tc, tc), tc)
        end
    end

    # Adding things preserves types.
    @test increment!!(zc, zc) isa Tt
    @test increment!!(zc, tc) isa Tt
    @test increment!!(tc, zc) isa Tt

    # Setting to zero equals zero.
    @test has_equal_data(set_to_zero!!(tc), z)
    if ismutabletype(P)
        @test set_to_zero!!(tc) === tc
    end
end

function test_numerical_testing_interface(p::P, t::T) where {P, T}
    @assert tangent_type(P) == T
    @test _scale(2.0, t) isa T
    @test _dot(t, t) isa Float64
    @test _dot(t, t) >= 0.0
    @test _dot(t, zero_tangent(p)) == 0.0
    @test _dot(t, increment!!(deepcopy(t), t)) ≈ 2 * _dot(t, t)
    @test _add_to_primal(p, t) isa P
    @test has_equal_data(_add_to_primal(p, zero_tangent(p)), p)
    @test _diff(p, p) isa T
    @test has_equal_data(_diff(p, p), zero_tangent(p))
end

end



module TestResources

using ..Taped
using ..Taped: CoDual, Tangent, MutableTangent, NoTangent, PossiblyUninitTangent

using DiffTests, LinearAlgebra, Random, Setfield


#
# Types used for testing purposes
#

function equal_field(a, b, f)
    (!isdefined(a, f) || !isdefined(b, f)) && return true
    return getfield(a, f) == getfield(b, f)
end

mutable struct Foo
    x::Real
end

Base.:(==)(a::Foo, b::Foo) = equal_field(a, b, :x)

struct StructFoo
    a::Real
    b::Vector{Float64}
    StructFoo(a::Float64, b::Vector{Float64}) = new(a, b)
    StructFoo(a::Float64) = new(a)
end

Base.:(==)(a::StructFoo, b::StructFoo) = equal_field(a, b, :a) && equal_field(a, b, :b)

mutable struct MutableFoo
    a::Float64
    b::AbstractVector
    MutableFoo(a::Float64, b::Vector{Float64}) = new(a, b)
    MutableFoo(a::Float64) = new(a)
end

Base.:(==)(a::MutableFoo, b::MutableFoo) = equal_field(a, b, :a) && equal_field(a, b, :b)

for T in [Foo, StructFoo, MutableFoo]
    @eval Taped._add_to_primal(p::$T, t) = Taped._containerlike_add_to_primal(p, t)
    @eval Taped._diff(p::$T, q::$T) = Taped._containerlike_diff(p, q)
end




#
# Functions for which rules are implemented. Useful for testing basic test infrastructure,
# and ensuring that any modifications to the interface do not prevent certain functions from
# having rules written for them. If a function is found for which the design of `rrule!!`
# doesn't permit a rule to be written, it should be added here to prevent future
# regressions. For example, `primitive_setfield!` was added because a particular iteration
# of the design did not allow the implementation of a correct rule.
# The hope is that this list of functions catches any issues early, before a large-scale
# re-write of the rules begins.
#

p_sin(x) = sin(x)

function Taped.rrule!!(::CoDual{typeof(p_sin)}, x::CoDual{Float64, Float64})
    p_sin_pb!!(ȳ::Float64, df, dx) = df, dx + ȳ * cos(primal(x))
    return CoDual(sin(primal(x)), zero(Float64)), p_sin_pb!!
end

p_mul(x, y) = x * y

function Taped.rrule!!(::CoDual{typeof(p_mul)}, x::CoDual{Float64}, y::CoDual{Float64})
    p_mul_pb!!(z̄, df, dx, dy) = df, dx + z̄ * primal(y), dy + z̄ * primal(x)
    return CoDual(primal(x) * primal(y), zero(Float64)), p_mul_pb!!
end

p_mat_mul!(C, A, B) = mul!(C, A, B)

function Taped.rrule!!(
    ::CoDual{typeof(p_mat_mul!)}, C::T, A::T, B::T
) where {T<:CoDual{Matrix{Float64}}}
    C_old = copy(C)
    function p_mat_mul_pb!!(C̄::Matrix{Float64}, df, _, Ā, B̄)
        Ā .+= C̄ * primal(B)'
        B̄ .+= primal(A)' * C̄
        primal(C) .= primal(C_old)
        shadow(C) .= shadow(C_old)
        return df, C̄, Ā, B̄
    end
    mul!(primal(C), primal(A), primal(B))
    shadow(C) .= 0
    return C, p_mat_mul_pb!!
end

p_setfield!(value, name::Symbol, x) = setfield!(value, name, x)

__replace_value(::T, v) where {T<:PossiblyUninitTangent} = T(v)

function __setfield!(value::MutableTangent, name, x)
    fields = value.fields
    new_fields = @set fields.$name = __replace_value(getfield(fields, name), x)
    value.fields = new_fields
    return x
end

function Taped.rrule!!(::CoDual{typeof(p_setfield!)}, value, name::CoDual{Symbol}, x)
    _name = primal(name)
    _value = primal(value)
    _dvalue = shadow(value)
    old_x = getfield(_value, _name)
    old_dx = getfield(_dvalue.fields, _name).tangent

    function p_setfield!_pb!!(dy, df, dvalue, dname, dx)

        # Add all increments to dx.
        dx = increment!!(dx, getfield(dvalue.fields, _name).tangent)
        dx = increment!!(dx, dy)

        # Restore old values.
        setfield!(primal(value), _name, old_x)
        __setfield!(shadow(value), _name, old_dx)

        return df, dvalue, dname, dx
    end

    y = CoDual(
        setfield!(_value, _name, primal(x)),
        __setfield!(_dvalue, _name, shadow(x)),
    )
    return y, p_setfield!_pb!!
end

for f in [p_sin, p_mul, p_mat_mul!, p_setfield!]
    @eval Taped.Umlaut.isprimitive(::Taped.RMC, ::typeof($f), x...) = true
end

const __A = randn(3, 3)

const PRIMITIVE_TEST_FUNCTIONS = Any[
    (p_sin, 5.0),
    (p_mul, 5.0, 4.0),
    (p_mat_mul!, randn(4, 5), randn(4, 3), randn(3, 5)),
    (p_mat_mul!, randn(3, 3), __A, __A),
    (p_setfield!, Foo(5.0), :x, 4.0),
    (p_setfield!, MutableFoo(5.0, randn(5)), :b, randn(6)),
]

#
# Tests for AD. There are not rules defined directly on these functions, and they require
# that most language primitives have rules defined.
#

test_sin(x) = sin(x)

test_cos_sin(x) = cos(sin(x))

test_isbits_multiple_usage(x::Float64) = Core.Intrinsics.mul_float(x, x)

function test_isbits_multiple_usage_2(x::Float64)
    y = Core.Intrinsics.mul_float(x, x)
    return Core.Intrinsics.mul_float(y, y)
end

function test_isbits_multiple_usage_3(x::Float64)
    y = sin(x)
    z = Core.Intrinsics.mul_float(y, y)
    a = Core.Intrinsics.mul_float(z, z)
    b = cos(a)
    return b
end

function test_isbits_multiple_usage_4(x::Float64)
    y = x > 0.0 ? cos(x) : sin(x)
    return Core.Intrinsics.mul_float(y, y)
end

function test_isbits_multiple_usage_5(x::Float64)
    y = Core.Intrinsics.mul_float(x, x)
    return x > 0.0 ? cos(y) : sin(y)
end

test_getindex(x::AbstractArray{<:Real}) = x[1]

function test_mutation!(x::AbstractVector{<:Real})
    x[1] = sin(x[2])
    return x[1]
end

function test_for_loop(x)
    for _ in 1:5
        x = sin(x)
    end
    return x
end

function test_while_loop(x)
    n = 3
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end

test_mutable_struct_basic(x) = Foo(x).x

test_mutable_struct_basic_sin(x) = sin(Foo(x).x)

function test_mutable_struct_setfield(x)
    foo = Foo(1.0)
    foo.x = x
    return foo.x
end

function test_mutable_struct(x)
    foo = Foo(x)
    foo.x = sin(foo.x)
    return foo.x
end

test_struct_partial_init(a::Float64) = StructFoo(a).a

test_mutable_partial_init(a::Float64) = MutableFoo(a).a

function test_naive_mat_mul!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}) where {T<:Real}
    for p in 1:size(C, 1)
        for q in 1:size(C, 2)
            C[p, q] = zero(T)
            for r in 1:size(A, 2)
                C[p, q] += A[p, r] * B[r, q]
            end
        end
    end
    return C
end

test_diagonal_to_matrix(D::Diagonal) = Matrix(D)

relu(x) = max(x, zero(x))

test_mlp(x, W1, W2) = W2 * relu.(W1 * x)

const TEST_FUNCTIONS = [
    (false, test_sin, 1.0),
    (false, test_cos_sin, 2.0),
    (false, test_isbits_multiple_usage, 5.0),
    (false, test_isbits_multiple_usage_2, 5.0),
    (false, test_isbits_multiple_usage_3, 4.1),
    (false, test_isbits_multiple_usage_4, 5.0),
    (false, test_isbits_multiple_usage_5, 4.1),
    (false, test_getindex, [1.0, 2.0]),
    (false, test_mutation!, [1.0, 2.0]),
    (false, test_while_loop, 2.0),
    (false, test_for_loop, 3.0),
    (false, test_mutable_struct_basic, 5.0),
    (false, test_mutable_struct_basic_sin, 5.0),
    (false, test_mutable_struct_setfield, 4.0),
    (false, test_mutable_struct, 5.0),
    (false, test_struct_partial_init, 3.5),
    (false, test_mutable_partial_init, 3.3),
    (false, test_naive_mat_mul!, randn(2, 1), randn(2, 1), randn(1, 1)),
    (false, (A, C) -> test_naive_mat_mul!(C, A, A), randn(2, 2), randn(2, 2)),
    (false, sum, randn(3)),
    (false, test_diagonal_to_matrix, Diagonal(randn(3))),
    (false, ldiv!, randn(2, 2), Diagonal(randn(2)), randn(2, 2)),
    (false, kron!, randn(4, 4), Diagonal(randn(2)), randn(2, 2)),
    (false, test_mlp, randn(5, 2), randn(7, 5), randn(3, 7)),
]

function value_dependent_control_flow(x, n)
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end

#
# This is a version of setfield! in which there is an issue with the address map.
# The method of setfield! is incorrectly implemented, so it errors. This is intentional,
# and is used to ensure that the tests correctly pick up on this mistake.
#

my_setfield!(args...) = setfield!(args...)

function _setfield!(value::MutableTangent, name, x)
    @set value.fields.$name = x
    return x
end

function Taped.rrule!!(::Taped.CoDual{typeof(my_setfield!)}, value, name, x)
    _name = primal(name)
    old_x = isdefined(primal(value), _name) ? getfield(primal(value), _name) : nothing
    function setfield!_pullback(dy, df, dvalue, ::NoTangent, dx)
        new_dx = increment!!(dx, getfield(dvalue.fields, _name).tangent)
        set_field_to_zero!!(dvalue, _name)
        new_dx = increment!!(new_dx, dy)
        old_x !== nothing && setfield!(primal(value), _name, old_x)
        return df, dvalue, NoTangent(), new_dx
    end
    y = Taped.CoDual(
        setfield!(primal(value), _name, primal(x)),
        _setfield!(shadow(value), _name, shadow(x)),
    )
    return y, setfield!_pullback
end

# Tests brought in from DiffTests.jl
const _n = rand()
const _x = rand(5, 5)
const _y = rand(26)
const _A = rand(5, 5)
const _B = rand(5, 5)
const _rng = Xoshiro(123456)

const DIFFTESTS_FUNCTIONS = vcat(
    tuple.(
        fill(false, length(DiffTests.NUMBER_TO_NUMBER_FUNCS)),
        DiffTests.NUMBER_TO_NUMBER_FUNCS,
        rand(_rng, length(DiffTests.NUMBER_TO_NUMBER_FUNCS)) .+ 1e-1,
    ),
    tuple.(
        fill(false, length(DiffTests.NUMBER_TO_ARRAY_FUNCS)),
        DiffTests.NUMBER_TO_ARRAY_FUNCS,
        [rand(_rng) + 1e-1 for _ in DiffTests.NUMBER_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS)),
        DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS,
        [rand(_rng, 5) .+ 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
        [rand(_rng) + 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.VECTOR_TO_NUMBER_FUNCS)),
        DiffTests.VECTOR_TO_NUMBER_FUNCS,
        [rand(_rng, 5) .+ 1e-1 for _ in DiffTests.VECTOR_TO_NUMBER_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.MATRIX_TO_NUMBER_FUNCS)),
        DiffTests.MATRIX_TO_NUMBER_FUNCS,
        [rand(_rng, 5, 5) .+ 1e-1 for _ in DiffTests.MATRIX_TO_NUMBER_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS)),
        DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS,
        [rand(_rng, 5, 5) .+ 1e-1 + I for _ in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS],
        [rand(_rng, 5, 5) .+ 1e-1 + I for _ in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS)),
        DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS,
        [rand(_rng, 5, 5) .+ 1e-1 for _ in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS],
        [rand(_rng, 5, 5) .+ 1e-1 for _ in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS],
        [rand(_rng, 5, 5) .+ 1e-1 for _ in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS)),
        DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS,
        [rand(_rng, 26) .+ 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
        [rand(_rng, 26) .+ 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.VECTOR_TO_VECTOR_FUNCS)),
        DiffTests.VECTOR_TO_VECTOR_FUNCS,
        [rand(_rng, 26) .+ 1e-1 for _ in DiffTests.VECTOR_TO_VECTOR_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.ARRAY_TO_ARRAY_FUNCS)),
        DiffTests.ARRAY_TO_ARRAY_FUNCS,
        [rand(_rng, 26) .+ 1e-1 for _ in DiffTests.ARRAY_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.MATRIX_TO_MATRIX_FUNCS)),
        DiffTests.MATRIX_TO_MATRIX_FUNCS,
        [rand(_rng, 5, 5) .+ 1e-1 for _ in DiffTests.MATRIX_TO_MATRIX_FUNCS],
    ),
)

end
