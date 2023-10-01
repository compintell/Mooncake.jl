function rrule!!(::CoDual{typeof(verify)}, args...)
    v = verify(map(primal, args)...)
    return CoDual(v, zero_tangent(v)), NoPullback()
end

function rrule!!(::CoDual{typeof(Umlaut.__new__)}, xs...)
    y = Umlaut.__new__(map(primal, xs)...)
    P = primal(xs[1])
    dy = build_tangent(P, map(shadow, xs[2:end])...)
    function __new__pullback(dy, d__new__, df, dxs...)
        new_dxs = map((x, y) -> increment!!(x, _value(y)), dxs, dy.fields)
        return d__new__, df, new_dxs...
    end
    function __new__pullback(dy::NamedTuple, d__new__, df, dxs...)
        new_dxs = map((x, y) -> increment!!(x, _value(y)), dxs, dy)
        return d__new__, df, new_dxs...
    end
    return CoDual(y, dy), __new__pullback
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

# Umlaut occassionally pushes `getindex` onto the tape.
# Easiest just to handle it like this.
# Might remove at a later date when `Umlaut.primitivize` works properly.
isprimitive(::RMC, ::typeof(getindex), ::Tuple, ::Int) = true
function rrule!!(::CoDual{typeof(getindex)}, x::CoDual{<:Tuple}, i::CoDual{Int})
    function getindex_pullback!!(dy, df, dx, ::NoTangent)
        dx = ntuple(n -> n == primal(i) ? increment!!(dx[n], dy) : dx[n], length(dx))
        return df, dx, NoTangent()
    end
    return CoDual(primal(x)[primal(i)], shadow(x)[primal(i)]), getindex_pullback!!
end
