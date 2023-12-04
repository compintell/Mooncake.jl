function rrule!!(::CoDual{typeof(verify)}, args...)
    return CoDual(verify(map(primal, args)...), zero_tangent(v)), NoPullback()
end

@generated function __new__pullback(dy, d__new__, df, dxs::Vararg{Any, N}) where {N}
    inc_exprs = map(n -> :(increment!!(dxs[$n], _value(fs[$n]))), 1:N)
    return quote
        fs = dy.fields
        return $(Expr(:tuple, :d__new__, :df, inc_exprs...))
    end
end

@generated function __new__pullback(
    dy::Union{Tuple, NamedTuple}, d__new__, df, dxs::Vararg{Any, N}
) where {N}
    inc_exprs = map(n -> :(increment!!(dxs[$n], _value(dy[$n]))), 1:N)
    return quote
        return $(Expr(:tuple, :d__new__, :df, inc_exprs...))
    end
end

@generated function rrule!!(
    ::CoDual{typeof(__new__)}, ::CoDual{Type{P}}, xs::Vararg{Any, N}
) where {P, N}
    return quote
        x_ps = map(primal, xs)
        y = $(Expr(:new, P, map(n -> :(x_ps[$n]), 1:N)...))
        dy = build_tangent(P, map(tangent, xs)...)
        return CoDual(y, dy), __new__pullback
    end
end

#
# Rules to avoid hitting limitations of Umlaut
# These should be removed at a later date.
#

isprimitive(::RMC, ::typeof(eltype), x) = true
function rrule!!(::CoDual{typeof(eltype)}, x)
    return CoDual(eltype(primal(x)), NoTangent()), NoPullback()
end

isprimitive(::RMC, ::typeof(Base.promote_op), x, S::Type...) = true
function rrule!!(::CoDual{typeof(Base.promote_op)}, args...)
    return CoDual(Base.promote_op(map(primal, args)...), NoTangent()), NoPullback()
end

isprimitive(::RMC, ::Core.Typeof(String), args...) = true
function rrule!!(::CoDual{Core.Typeof(String)}, args::CoDual...)
    s = String(map(primal, args)...)
    return CoDual(s, zero_tangent(s)), NoPullback()
end

#
# Rules to avoid / cope with Umlaut internals.
# We _might_ remove these at a later date if they turn out not to be needed.
#

function rrule!!(::CoDual{typeof(Umlaut.check_variable_length)}, args::Vararg{Any, N}) where {N}
    v = Umlaut.check_variable_length(map(primal, args)...)
    return CoDual(v, zero_tangent(v)), NoPullback()
end

# This is the thing that Umlaut uses in order to splat. Must be a primitive.
isprimitive(::RMC, ::typeof(__to_tuple__), x) = true
function rrule!!(::CoDual{typeof(__to_tuple__)}, x::CoDual{<:Tuple})
    __to_tuple_pb!!(dy, df, dx) = df, increment!!(dx, dy)
    return x, __to_tuple_pb!!
end
function rrule!!(::CoDual{typeof(__to_tuple__)}, x::CoDual{<:NamedTuple{A}}) where {A}
    __to_tuple_named_tuple_pb!!(dy, df, dx) = df, increment!!(dx, NamedTuple{A}(dy))
    return CoDual(Tuple(primal(x)), Tuple(tangent(x))), __to_tuple_named_tuple_pb!!
end
function rrule!!(::CoDual{typeof(__to_tuple__)}, x::CoDual{<:Vector, <:Vector{T}}) where {T}
    __to_tuple_vec_pb!!(dy, df, dx) = df, increment!!(dx, T[a for a in dy])
    return CoDual(__to_tuple__(primal(x)), __to_tuple__(tangent(x))), __to_tuple_vec_pb!!
end
function rrule!!(::CoDual{typeof(__to_tuple__)}, x::CoDual{Int})
    return zero_codual((primal(x), )), NoPullback()
end
function rrule!!(::CoDual{typeof(__to_tuple__)}, x::CoDual{Core.SimpleVector})
    function __to_tuple_svec_pb!!(dy, df, dx)
        return df, increment!!(dx, Any[a for a in dy])
    end
    y = CoDual(__to_tuple__(primal(x)), __to_tuple__(tangent(x)))
    return y, __to_tuple_svec_pb!!
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:umlaut_internals_rules})
    test_cases = Any[
        (false, :stability, nothing, __new__, UnitRange{Int}, 5, 9),
        (false, :none, nothing, __new__, TestResources.StructFoo, 5.0, randn(4)),
        (false, :none, nothing, __new__, TestResources.MutableFoo, 5.0, randn(5)),
        (false, :none, nothing, __new__, TestResources.StructFoo, 5.0),
        (false, :none, nothing, __new__, TestResources.MutableFoo, 5.0),
        (
            false,
            :stability,
            nothing,
            __new__,
            TestResources.TypeStableMutableStruct{Vector{Float64}},
            5.0,
            randn(5),
        ),
        (
            false,
            :stability,
            nothing,
            __new__,
            TestResources.TypeStableMutableStruct{Vector{Float64}},
            5.0,
        ),
        (false, :stability, nothing, __new__, NamedTuple{(), Tuple{}}),
        (
            false,
            :stability,
            nothing,
            __new__,
            NamedTuple{(:a, :b), Tuple{Float64, Float64}},
            5.0,
            4.0,
        ),
        (false, :stability, nothing, __new__, Tuple{Float64, Float64}, 5.0, 4.0),

        # Splatting primitives:
        (false, :stability, nothing, __to_tuple__, (5.0, 4)),
        (false, :stability, nothing, __to_tuple__, (a=5.0, b=4)),
        (false, :stability, nothing, __to_tuple__, 5),
        (false, :none, nothing, __to_tuple__, svec(5.0)),
        (false, :none, nothing, __to_tuple__, [5.0, 4.0]),

        # Umlaut limitations:
        (false, :none, nothing, eltype, randn(5)),
        (false, :none, nothing, eltype, transpose(randn(4, 5))),
        (false, :none, nothing, Base.promote_op, transpose, Float64),
        (true, :none, nothing, String, lazy"hello world"),
    ]
    memory = Any[]
    return test_cases, memory
end

__multiarg_fn(x) = only(x)
__multiarg_fn(x, y) = only(x) + only(y)
__multiarg_fn(x, y, z) = only(x) + only(y) + only(z)

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:umlaut_internals_rules})


    test_cases = Any[
        [false, nothing, x -> __multiarg_fn(x...), 1],
        [false, nothing, x -> __multiarg_fn(x...), [1.0, 2.0]],
        [false, nothing, x -> __multiarg_fn(x...), [5.0, 4]],
        [false, nothing, x -> __multiarg_fn(x...), (5.0, 4)],
        [false, nothing, x -> __multiarg_fn(x...), (a=5.0, b=4)],
        [false, nothing, x -> __multiarg_fn(x...), svec(5.0, 4.0)],
    ]
    memory = Any[]
    return test_cases, memory
end