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

function tuple_map(f::F, x::NamedTuple{names}) where {F, names}
    return NamedTuple{names}(tuple_map(f, Tuple(x)))
end

function tuple_map(f::F, x::NamedTuple{names}, y::NamedTuple{names}) where {F, names}
    return NamedTuple{names}(tuple_map(f, Tuple(x), Tuple(y)))
end

function is_vararg_sig_and_sparam_names(sig)
    world = Base.get_world_counter()
    min = Base.RefValue{UInt}(typemin(UInt))
    max = Base.RefValue{UInt}(typemax(UInt))
    ms = Base._methods_by_ftype(sig, nothing, -1, world, true, min, max, Ptr{Int32}(C_NULL))::Vector
    m = only(ms).method
    return m.isva, sparam_names(m)
end

function sparam_names(m::Core.Method)::Vector{Symbol}
    whereparams = ExprTools.where_parameters(m.sig)
    whereparams === nothing && return Symbol[]
    return map(whereparams) do name
        name isa Symbol && return name
        Meta.isexpr(name, :(<:)) && return name.args[1]
        Meta.isexpr(name, :(>:)) && return name.args[1]
        error("unrecognised type param $name")
    end
end
