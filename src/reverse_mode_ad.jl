struct ReverseModeADContext <: TapedContext end

const RMC = ReverseModeADContext

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
shadow(x::CoDual) = x.dx
Base.copy(x::CoDual) = CoDual(copy(primal(x)), copy(shadow(x)))

"""
    uninit_codual(x)

See implementation for details, as this function is subject to change.
"""
uninit_codual(x) = CoDual(x, uninit_tangent(x))

set_shadow!!(x::CoDual, dx) = CoDual(primal(x), increment!!(set_to_zero!!(shadow(x)), dx))

function verify_codual_type(::CoDual{P, T}) where {P, T}
    Tt = tangent_type(P)
    if Tt !== T
        throw(error("for primal of type $P, expected tangent of type $Tt, but found $T"))
    end
end

struct NoPullback end

@inline (::NoPullback)(dy, dx...) = dx

might_be_active(args) = any(might_be_active ∘ typeof, args)

isprimitive(::RMC, args...) = might_be_active(args) ? false : true
isprimitive(::RMC, ::typeof(Umlaut.__new__), T, x...) = true
isprimitive(::RMC, ::typeof(Umlaut.__foreigncall__), args...) = true
isprimitive(::RMC, ::typeof(__intrinsic__), args...) = true
isprimitive(::RMC, ::Core.Builtin, x...) = true

"""
    rebind(x)

Primal evaluation is equivalent to the identity function. However, the `rrule!!` ensures
that something sensible happens with the tangent. Not to be used by downstream users.
Is basically a hack to ensure that double counting doesn't occur.

TODO: improve docstring + explain / understand semantics better.
"""
rebind(x) = x

rebind_pb!!(ȳ, f̄, x̄) = f̄, increment!!(x̄, ȳ)
function rrule!!(::CoDual{typeof(rebind)}, x::CoDual)
    return CoDual(primal(x), rebind_tangent(shadow(x))), rebind_pb!!
end

isprimitive(::RMC, ::typeof(rebind), x) = true

#
# LinearAlgebra
#

function rrule!!(::CoDual{typeof(LinearAlgebra.chkstride1)}, args...)
    return CoDual(LinearAlgebra.chkstride1(args...), NoTangent()), NoPullback()
end

isprimitive(::RMC, ::Type, ::TypeVar, ::Type) = true
function rrule!!(x::CoDual{<:Type}, y::CoDual{<:TypeVar}, z::CoDual{<:Type})
    return CoDual(primal(x)(primal(y), primal(z)), NoTangent()), NoPullback()
end
