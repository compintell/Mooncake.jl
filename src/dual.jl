struct Dual{P, T}
    x::P
    dx::T
end

function Dual(x::P, dx::T) where {P,T}
    if T != tangent_type(P)
        throw(ArgumentError("Tried to build a `Dual(x, dx)` with `x::$P` and `dx::$T` but the correct tangent type is `$(tangent_type(P))`")
    end
    return Dual{P,T}(x, dx)
end

primal(x::Dual) = x.x
tangent(x::Dual) = x.dx
Base.copy(x::Dual) = Dual(copy(primal(x)), copy(tangent(x)))
_copy(x::P) where {P<:Dual} = x
