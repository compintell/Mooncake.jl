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

function __new__pullback(
    dy::Union{Tuple, NamedTuple}, d__new__, df, dxs::Vararg{Any, N}
) where {N}
    new_dxs = map((x, y) -> increment!!(x, _value(y)), dxs, dy)
    return d__new__, df, new_dxs...
end

@generated function rrule!!(
    ::CoDual{typeof(Umlaut.__new__)}, ::CoDual{Type{P}}, xs::Vararg{Any, N}
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
