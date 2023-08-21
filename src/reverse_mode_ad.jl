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

function verify_codual_type(::CoDual{P, T}) where {P, T}
    Tt = tangent_type(P)
    if Tt !== T
        throw(error("for primal of type $P, expected tangent of type $Tt, but found $T"))
    end
end

struct NoPullback end

@inline (::NoPullback)(dy, dx...) = dx

function isprimitive(::RMC, args...)
    return any(might_be_active, map(Core.Typeof , args)) ? false : true
end
isprimitive(::RMC, ::typeof(Umlaut.__new__), T, x...) = true
isprimitive(::RMC, ::typeof(Umlaut.__foreigncall__), args...) = true
isprimitive(::RMC, ::typeof(__intrinsic__), args...) = true
isprimitive(::RMC, ::Core.Builtin, x...) = true


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
