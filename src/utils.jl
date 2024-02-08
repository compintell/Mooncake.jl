"""
    _typeof(x)

Central definition of typeof, which is specific to the use-required in this package.
"""
_typeof(x) = Base._stable_typeof(x)
_typeof(x::Tuple) = Tuple{map(_typeof, x)...}
_typeof(x::NamedTuple{names}) where {names} = NamedTuple{names, _typeof(Tuple(x))}
