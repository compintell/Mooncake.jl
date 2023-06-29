increment!!(::NoTangent, ::NoTangent) = NoTangent()
struct Tangent{Tfields<:NamedTuple}
    fields::Tfields
end

Base.:(==)(x::Tangent, y::Tangent) = x.fields == y.fields

mutable struct MutableTangent{Tfields<:NamedTuple}
    fields::Tfields
end

Base.:(==)(x::MutableTangent, y::MutableTangent) = x.fields == y.fields

"""
    increment!!(x::T, y::T) where {T}

Add `x` to `y`. If `ismutabletype(T)`, then `increment!!(x, y) === x` must hold.
That is, `increment!!` will mutate `x`.
This must apply recursively if `T` is a composite type whose fields are mutable.
"""
increment!!(x::T, y::T) where {T<:Base.IEEEFloat} = x + y
increment!!(x::T, y::T) where {T<:Array} = x .+= y
increment!!(x::T, y::T) where {T<:Tuple} = map(increment!!, x, y)
increment!!(x::T, y::T) where {T<:NamedTuple} = map(increment!!, x, y)
increment!!(x::T, y::T) where {T<:Tangent} = Tangent(increment!!(x.fields, y.fields))
function increment!!(x::T, y::T) where {T<:MutableTangent}
    x.fields = increment!!(x.fields, y.fields)
    return x
end

"""
    increment_field!!(x::T, y::V, f) where {T, V}

`increment!!` the field `f` of `x` by `y`, and return the updated `x`.
"""
increment_field!!(x::NamedTuple, y, f) = @set x.$f = increment!!(getfield(x, f), y)
function increment_field!!(x::Tuple, y, i::Int)
    return ntuple(n -> n == i ? increment!!(x[n], y) : x[n], length(x))
end
increment_field!!(x::Tangent, y, f) = Tangent(increment_field!!(x.fields, y, f))
function increment_field!!(x::MutableTangent, y, f)
    setfield!(x, :fields, increment_field!!(x.fields, y, f))
    return x
end

"""
    randn_tangent(rng::AbstractRNG, x)

Generate a randomly-chosen tangent to `x`.
"""
randn_tangent(::AbstractRNG, ::NoTangent) = NoTangent()
randn_tangent(rng::AbstractRNG, ::T) where {T<:Base.IEEEFloat} = randn(rng, T)
function randn_tangent(rng::AbstractRNG, x::Union{Array, Tuple, NamedTuple})
    return map(x -> randn_tangent(rng, x), x)
end
function randn_tangent(rng::AbstractRNG, x::T) where {T<:Union{Tangent, MutableTangent}}
    return T(randn_tangent(rng, x.fields))
end

"""
    set_to_zero!!(x)

Set `x` to its zero element (`x` should be a tangent, so the zero must exist).
"""
set_to_zero!!(::NoTangent) = NoTangent()
set_to_zero!!(x::Base.IEEEFloat) = zero(x)
set_to_zero!!(x::Union{Tuple, NamedTuple}) = map(set_to_zero!!, x)
set_to_zero!!(x::Array) = map!(set_to_zero!!, x, x)
set_to_zero!!(x::Tangent) = Tangent(set_to_zero!!(x.fields))
function set_to_zero!!(x::MutableTangent)
    x.fields = set_to_zero!!(x.fields)
    return x
end

"""
    set_field_to_zero!!(x, f)

Set the field `f` of `x` to zero -- `f` can be an integer if `x` is a `Tuple`.
"""
set_field_to_zero!!(x::NamedTuple, f) = @set x.$f = set_to_zero!!(getfield(x, f))
function set_field_to_zero!!(x::Tuple, i::Int)
    ntuple(n -> n == i ? set_to_zero!!(x[n]) : x[n], length(x))
end
set_field_to_zero!!(x::Tangent, f) = Tangent(set_field_to_zero!!(x.fields, f))
function set_field_to_zero!!(x::MutableTangent, f)
    x.fields = set_field_to_zero!!(x.fields, f)
    return x
end
