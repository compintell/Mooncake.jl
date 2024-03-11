"""
    _typeof(x)

Central definition of typeof, which is specific to the use-required in this package.
"""
_typeof(x) = Base._stable_typeof(x)
_typeof(x::Tuple) = Tuple{map(_typeof, x)...}
_typeof(x::NamedTuple{names}) where {names} = NamedTuple{names, _typeof(Tuple(x))}

"""
    tuple_map(f::F, x::Tuple) where {F}

This function is semantically equivalent to `map(f, x)`, but always specialises on all of
the element types of `x`, regardless the length of `x`. This contrasts with `map`, in which
the number of element types specialised upon is a fixed constant in the compiler.

As a consequence, if `x` is very long, this function may have very large compile times.

    tuple_map(f::F, x::Tuple, y::Tuple) where {F}

Binary extension of `tuple_map`. Equivalent to `map(f, x, y`, but guaranteed to specialise
on all element types of `x` and `y`.
"""
@inline @generated function tuple_map(f::F, x::Tuple) where {F}
    return Expr(:call, :tuple, map(n -> :(f(getfield(x, $n))), eachindex(x.parameters))...)
end

@inline @generated function tuple_map(f::F, x::Tuple, y::Tuple) where {F}
    return Expr(:call, :tuple, map(n -> :(f(getfield(x, $n), getfield(y, $n))), eachindex(x.parameters))...)
end
