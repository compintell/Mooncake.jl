struct ReverseModeADContext <: TapedContext end

const RMC = ReverseModeADContext

struct CoDual{Tx, Tdx}
    x::Tx
    dx::Tdx
end

# Always sharpen the first thing if it's a type, in order to preserve dispatch possibility.
CoDual(::Type{P}, dx::NoTangent) where {P} = CoDual{Type{P}, NoTangent}(P, dx)

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




#
# LinearAlgebra
#

function rrule!!(::CoDual{typeof(LinearAlgebra.chkstride1)}, args...)
    return CoDual(LinearAlgebra.chkstride1(args...), NoTangent()), NoPullback()
end
