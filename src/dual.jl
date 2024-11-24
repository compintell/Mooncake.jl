struct Dual{P,T}
    primal::P
    tangent::T
end

primal(x::Dual) = x.primal
tangent(x::Dual) = x.tangent
Base.copy(x::Dual) = Dual(copy(primal(x)), copy(tangent(x)))
_copy(x::P) where {P<:Dual} = x

zero_dual(x) = Dual(x, zero_tangent(x))

dual_type(::Type{P}) where {P} = Dual{P,tangent_type(P)}
