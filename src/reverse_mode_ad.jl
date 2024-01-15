struct CoDual{Tx, Tdx}
    x::Tx
    dx::Tdx
end

# Always sharpen the first thing if it's a type, in order to preserve dispatch possibility.
function CoDual(x::Type{P}, dx::NoTangent) where {P}
    if @isdefined(P)
        return CoDual{Type{P}, NoTangent}(P, dx)
    else
        return CoDual{typeof(x), NoTangent}(x, dx)
    end
end

primal(x::CoDual) = x.x
tangent(x::CoDual) = x.dx
Base.copy(x::CoDual) = CoDual(copy(primal(x)), copy(tangent(x)))

"""
    zero_codual(x)

Equivalent to `CoDual(x, zero_tangent(x))`.
"""
zero_codual(x) = CoDual(x, zero_tangent(x))

"""
    uninit_codual(x)

See implementation for details, as this function is subject to change.
"""
uninit_codual(x) = CoDual(x, uninit_tangent(x))

"""
    randn_codual(rng::AbstractRNG, x)

Equivalent to `CoDual(x, randn_tangent(rng, x))`.
"""
randn_codual(rng::AbstractRNG, x) = CoDual(x, randn_tangent(rng, x))

"""
    codual_type(P::Type)

Shorthand for `CoDual{P, tangent_type(P}}` when `P` is concrete, equal to `CoDual` if not.
"""
function codual_type(::Type{P}) where {P}
    P == DataType && return CoDual
    return isconcretetype(P) ? CoDual{P, tangent_type(P)} : CoDual
end

set_tangent!!(x::CoDual, dx) = CoDual(primal(x), increment!!(set_to_zero!!(tangent(x)), dx))

function verify_codual_type(::CoDual{P, T}) where {P, T}
    Tt = tangent_type(P)
    if Tt !== T
        throw(error("for primal of type $P, expected tangent of type $Tt, but found $T"))
    end
end

struct NoPullback end

@inline (::NoPullback)(dy, dx...) = dx

might_be_active(args) = any(might_be_active ∘ typeof, args)
