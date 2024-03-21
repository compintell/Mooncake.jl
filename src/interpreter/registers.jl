"""
    AugmentedRegister(codual::CoDual, tangent_stack)

A wrapper data structure for bundling together a codual and a tangent stack. These appear
in the code associated to active values in the primal.

For example, a statment in the primal such as
```julia
%5 = sin(%4)::Float64
```
which provably returns a `Float64` in the primal, would return an `register_type(Float64)`
in the forwards-pass, where `register_type` will return an `AugmentedRegister` when the
primal type is `Float64`.
"""
struct AugmentedRegister{T<:CoDual, V}
    codual::T
    tangent_ref::V
end

@inline primal(reg::AugmentedRegister) = primal(reg.codual)

"""
    register_type(::Type{P}) where {P}

If `P` is the type associated to a primal register, the corresponding register in the
forwards-pass must be a `register_type(P)`.
"""
function register_type(::Type{P}) where {P}
    P == DataType && return Any
    P == UnionAll && return Any
    P isa Union && return __union_register_type(P)
    if isconcretetype(P)
        return AugmentedRegister{codual_type(P), tangent_ref_type_ub(P)}
    else
        return AugmentedRegister
    end
end

# Specialised method for unions.
function __union_register_type(::Type{P}) where {P}
    if P isa Union
        CC.tmerge(AugmentedRegister{codual_type(P.a)}, __union_register_type(P.b))
    else
        return AugmentedRegister{codual_type(P)}
    end
end
