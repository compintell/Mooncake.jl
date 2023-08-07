"""
    NoTangent

The type in question has no meaningful notion of a tangent space.
Generally, you shouldn't use this -- just let the default recursive tangent
construction work.
You might need to use this for `primitive type`s though.
"""
struct NoTangent end

"""
    PossiblyUninitTangent{T}

Represents a `T` which maybe or may not be present. Does not distinguish between 0 and
not being present.
"""
struct PossiblyUninitTangent{T}
    tangent::T
    PossiblyUninitTangent{T}(tangent::T) where {T} = new{T}(tangent)
    PossiblyUninitTangent{T}() where {T} = new{T}()
end

@inline PossiblyUninitTangent(tangent::T) where {T} = PossiblyUninitTangent{T}(tangent)
@inline PossiblyUninitTangent(T::Type) = PossiblyUninitTangent{T}()

const __PUT = PossiblyUninitTangent

@inline is_init(t::PossiblyUninitTangent) = isdefined(t, :tangent)

function Base.:(==)(t::PossiblyUninitTangent{T}, s::PossiblyUninitTangent{T}) where {T}
    is_init(t) && is_init(s) && return t.tangent == s.tangent
    is_init(t) && !is_init(s) && return false
    !is_init(t) && is_init(s) && return false
    return true
end

function Base.:+(t::PossiblyUninitTangent{T}, s::PossiblyUninitTangent{T}) where {T}
    is_init(t) && is_init(s) && return PossiblyUninitTangent{T}(t.tangent + s.tangent)
    is_init(t) && !is_init(s) && return t
    !is_init(t) && is_init(s) && return s
    return t
end

_wrap_type(::Type{T}) where {T} = PossiblyUninitTangent{T}

_wrap_field(::Type{Q}, x::T) where {Q, T} = PossiblyUninitTangent{Q}(x)
_wrap_field(x::T) where {T} = _wrap_field(T, x)

struct Tangent{Tfields<:NamedTuple}
    fields::Tfields
end

Base.:(==)(x::Tangent, y::Tangent) = x.fields == y.fields

mutable struct MutableTangent{Tfields<:NamedTuple}
    fields::Tfields
end

Base.:(==)(x::MutableTangent, y::MutableTangent) = x.fields == y.fields

function build_tangent(::Type{P}, fields...) where {P}
    tangent_types = map(P -> PossiblyUninitTangent{tangent_type(P)}, fieldtypes(P))
    tangent_values = ntuple(length(tangent_types)) do n
        n <= length(fields) ? tangent_types[n](fields[n]) : tangent_types[n]()
    end
    return tangent_type(P)(NamedTuple{fieldnames(P)}(tangent_values))
end

_value(v::PossiblyUninitTangent) = v.tangent
_value(v) = v

"""
    tangent_type(T)

The type used to represent the tangent of an object of type `T` must be unique, and
determined entirely by its type.
"""
tangent_type(T)

function tangent_type(x)
    throw(error("$x is not a type. Perhaps you meant typeof(x)?"))
end

# This is essential for DataType, as the recursive definition always recurses infinitely,
# because one of the fieldtypes is itself always a DataType. In particular, we'll always
# eventually hit `Any`, whose `super` field is `Any`.
# This makes it clear that we can't recursively construct tangents for data structures which
# refer to themselves...
tangent_type(::Type{<:Type}) = NoTangent

tangent_type(::Type{Ptr{P}}) where {P} = Ptr{tangent_type(P)}

tangent_type(::Type{<:Ptr}) = NoTangent

tangent_type(::Type{Bool}) = NoTangent

tangent_type(::Type{Char}) = NoTangent

tangent_type(::Type{Union{}}) = NoTangent

tangent_type(::Type{Symbol}) = NoTangent

tangent_type(::Type{P}) where {P<:Union{UInt8, UInt16, UInt32, UInt64, UInt128}} = NoTangent

tangent_type(::Type{P}) where {P<:Union{Int8, Int16, Int32, Int64, Int128}} = NoTangent

tangent_type(::Type{<:Core.Builtin}) = NoTangent

tangent_type(::Type{P}) where {P<:IEEEFloat} = P

tangent_type(::Type{<:Core.LLVMPtr}) = NoTangent

tangent_type(::Type{String}) = NoTangent

tangent_type(::Type{<:Array{P, N}}) where {P, N} = Array{tangent_type(P), N}

tangent_type(::Type{<:Array{P, N} where {P}}) where {N} = Array{Any, N}

tangent_type(::Type{<:MersenneTwister}) = NoTangent

tangent_type(::Type{Core.TypeName}) = NoTangent

function tangent_type(::Type{T}) where {T<:Tuple}
    if isconcretetype(T)
        return Tuple{map(tangent_type, fieldtypes(T))...}
    else
        return Tuple
    end
end

tangent_type(::Type{NamedTuple{N, T}}) where {N, T<:Tuple} = NamedTuple{N, tangent_type(T)}

function tangent_type(P::Type)

    # This method can only handle struct types. Tell user to implement tangent type
    # directly for primitive types.
    isprimitivetype(P) && throw(error(
        "$P is a primitive type. Implement a method of `tangent_type` for it."
    ))

    # If the type is a Union, then take the union type of its arguments.
    P isa Union && return Union{tangent_type(P.a), tangent_type(P.b)}

    # If the type is itself abstract, it's tangent could be anything.
    # The same goes for if the type has any undetermined type parameters.
    (isabstracttype(P) || !isconcretetype(P)) && return Any

    # Derive tangent type.
    tangent_field_names = fieldnames(P)
    tangent_field_types = map(t -> _wrap_type(tangent_type(t)), fieldtypes(P))
    T = ismutabletype(P) ? MutableTangent : Tangent
    return T{NamedTuple{tangent_field_names, Tuple{tangent_field_types...}}}
end

"""
    zero_tangent(x)

Returns the unique zero element of the tangent space of `x`.
It is an error for the zero element of the tangent space of `x` to be represented by
anything other than that which this function returns.
"""
zero_tangent(x)
zero_tangent(::Union{Int8, Int16, Int32, Int64, Int128}) = NoTangent()
zero_tangent(x::IEEEFloat) = zero(x)
zero_tangent(x::Array{P, N}) where {P, N} = map(zero_tangent, x)
zero_tangent(x::Union{Tuple, NamedTuple}) = map(zero_tangent, x)
@noinline function zero_tangent(x::P) where {P}

    tangent_type(P) == NoTangent && return NoTangent()

    # This method can only handle struct types. Tell user to implement tangent type
    # directly for primitive types.
    isprimitivetype(P) && throw(error(
        "$P is a primitive type. Implement a method of `zero_tangent` for it."
    ))

    # Derive zero tangent. Tangent types of fields, and types of zeros need only agree
    # if field types are concrete.
    tangent_field_names = fieldnames(P)
    tangent_field_types = map(tangent_type, fieldtypes(P))
    tangent_field_zeros = map(tangent_field_names, tangent_field_types) do field_name, T
        if isdefined(x, field_name)
            return PossiblyUninitTangent{T}(zero_tangent(getfield(x, field_name)))
        else
            return PossiblyUninitTangent{T}()
        end
    end
    wrapped_tangent_field_types = map(_wrap_type, tangent_field_types)
    Tbacking = NamedTuple{tangent_field_names, Tuple{wrapped_tangent_field_types...}}
    backing = Tbacking(tangent_field_zeros)
    T = ismutabletype(P) ? MutableTangent : Tangent
    return T{Tbacking}(backing)
end

"""
    randn_tangent(rng::AbstractRNG, x::T) where {T}

Required for testing.
Generate a randomly-chosen tangent to `x`.
"""
randn_tangent(::AbstractRNG, ::NoTangent) = NoTangent()
randn_tangent(rng::AbstractRNG, ::T) where {T<:IEEEFloat} = randn(rng, T)
randn_tangent(rng::AbstractRNG, x::Array{T}) where {T} = map(x -> randn_tangent(rng, x), x)
function randn_tangent(rng::AbstractRNG, x::Union{Tuple, NamedTuple})
    return map(x -> randn_tangent(rng, x), x)
end
function randn_tangent(rng::AbstractRNG, x::T) where {T<:Union{Tangent, MutableTangent}}
    return T(randn_tangent(rng, x.fields))
end
function randn_tangent(rng::AbstractRNG, x::P) where {P}

    # If `P` doesn't have a tangent space, always return `NoTangent()`.
    tangent_type(P) === NoTangent && return NoTangent()

    # This method can only handle struct types. Tell user to implement tangent type
    # directly for primitive types.
    isprimitivetype(P) && throw(error(
        "$P is a primitive type. Implement a method of `randn_tangent` for it."
    ))

    # Assume `P` is a generic struct type, and derive the tangent recursively.
    tangent_field_names = fieldnames(P)
    tangent_field_types = map(tangent_type, fieldtypes(P))
    tangent_field_vals = map(tangent_field_names, tangent_field_types) do field_name, T
        if isdefined(x, field_name)
            return PossiblyUninitTangent{T}(randn_tangent(rng, getfield(x, field_name)))
        else
            return PossiblyUninitTangent{T}()
        end
    end
    wrapped_tangent_field_types = map(_wrap_type, tangent_field_types)
    Tbacking = NamedTuple{tangent_field_names, Tuple{wrapped_tangent_field_types...}}
    backing = Tbacking(tangent_field_vals)
    T = ismutabletype(P) ? MutableTangent : Tangent
    return T{Tbacking}(backing)
end

"""
    increment!!(x::T, y::T) where {T}

Add `x` to `y`. If `ismutabletype(T)`, then `increment!!(x, y) === x` must hold.
That is, `increment!!` will mutate `x`.
This must apply recursively if `T` is a composite type whose fields are mutable.
"""
increment!!(::NoTangent, ::NoTangent) = NoTangent()
increment!!(x::T, y::T) where {T<:IEEEFloat} = x + y
increment!!(x::Ptr{T}, y::Ptr{T}) where {T} = x === y ? x : throw(error("eurgh"))
function increment!!(x::T, y::T) where {T<:Array}
    x === y || map!(increment!!, x, x, y)
    return x
end
increment!!(x::T, y::T) where {T<:Tuple} = map(increment!!, x, y)
increment!!(x::T, y::T) where {T<:NamedTuple} = map(increment!!, x, y)
function increment!!(x::T, y::T) where {T<:PossiblyUninitTangent}
    is_init(x) && is_init(y) && return T(increment!!(x.tangent, y.tangent))
    !is_init(x) && !is_init(y) && return x
    is_init(x) && !is_init(y) && error("x is initialised, but y is not")
    !is_init(x) && is_init(y) && error("x is not initialised, but y is")
end
increment!!(x::T, y::T) where {T<:Tangent} = Tangent(increment!!(x.fields, y.fields))
function increment!!(x::T, y::T) where {T<:MutableTangent}
    x === y && return x
    x.fields = increment!!(x.fields, y.fields)
    return x
end

"""
    set_to_zero!!(x)

Set `x` to its zero element (`x` should be a tangent, so the zero must exist).
"""
set_to_zero!!(::NoTangent) = NoTangent()
set_to_zero!!(x::Base.IEEEFloat) = zero(x)
set_to_zero!!(x::Union{Tuple, NamedTuple}) = map(set_to_zero!!, x)
set_to_zero!!(x::Array) = map!(set_to_zero!!, x, x)
function set_to_zero!!(x::T) where {T<:PossiblyUninitTangent}
    return is_init(x) ? T(set_to_zero!!(x.tangent)) : x
end
set_to_zero!!(x::Tangent) = Tangent(set_to_zero!!(x.fields))
function set_to_zero!!(x::MutableTangent)
    x.fields = set_to_zero!!(x.fields)
    return x
end

"""
    increment_field!!(x::T, y::V, f) where {T, V}

`increment!!` the field `f` of `x` by `y`, and return the updated `x`.
"""
increment_field!!(::NoTangent, ::NoTangent, f) = NoTangent()
increment_field!!(x::NamedTuple, y, f) = @set x.$f = increment!!(getfield(x, f), y)
function increment_field!!(x::Tuple, y, i::Int)
    return ntuple(n -> n == i ? increment!!(x[n], y) : x[n], length(x))
end
function increment_field!!(x::Tangent{T}, y, f) where {T}
    y isa NoTangent && return x
    return Tangent(increment_field!!(x.fields, fieldtype(T, f)(y), f))
end
function increment_field!!(x::MutableTangent{T}, y, f) where {T}
    y isa NoTangent && return x
    setfield!(x, :fields, increment_field!!(x.fields, fieldtype(T, f)(y), f))
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

"""
    _scale(a::Float64, t::T) where {T}

Required for testing.
Should be defined for all standard tangent types.

Multiply tangent `t` by scalar `a`. Always possible because any given tangent type must
correspond to a vector field. Not using `*` in order to avoid piracy.
"""
_scale(::Float64, ::NoTangent) = NoTangent()
_scale(a::Float64, t::T) where {T<:IEEEFloat} = T(a * t)
_scale(a::Float64, t::Array) = map(Base.Fix1(_scale, a), t)
_scale(a::Float64, t::Union{Tuple, NamedTuple}) = map(Base.Fix1(_scale, a), t)
function _scale(a::Float64, t::PossiblyUninitTangent{T}) where {T}
    return if is_init(t)
        return PossiblyUninitTangent{T}(_scale(a, t.tangent))
    else
        return PossiblyUninitTangent{T}()
    end
end
_scale(a::Float64, t::T) where {T<:Union{Tangent, MutableTangent}} = T(_scale(a, t.fields))

"""
    _dot(t::T, s::T)::Float64 where {T}

Required for testing.
Should be defined for all standard tangent types.

Inner product between tangents `t` and `s`. Must return a `Float64`.
Always available because all tangent types correspond to finite-dimensional vector spaces.
"""
_dot(::NoTangent, ::NoTangent) = 0.0
_dot(t::T, s::T) where {T<:IEEEFloat} = Float64(t * s)
_dot(t::T, s::T) where {T<:Array} = sum(map(_dot, t, s))
_dot(t::T, s::T) where {T<:Union{Tuple, NamedTuple}} = sum(map(_dot, t, s); init=0.0)
function _dot(t::T, s::T) where {T<:PossiblyUninitTangent}
    is_init(t) && is_init(s) && return _dot(t.tangent, s.tangent)
    return 0.0
end
function _dot(t::T, s::T) where {T<:Union{Tangent, MutableTangent}}
    return sum(map(_dot, t.fields, s.fields); init=0.0)
end

"""
    _add_to_primal(p::P, t::T) where {P, T}

Required for testing.
_Not_ currently defined by default.
`_containerlike_add_to_primal` is potentially what you want to target when implementing for
a particular primal-tangent pair.

Adds `t` to `p`, returning a `P`. It must be the case that `tangnet_type(P) == T`.
"""
_add_to_primal(x, ::NoTangent) = x
_add_to_primal(x::T, t::T) where {T<:IEEEFloat} = x + t
_add_to_primal(x::Array, t::Array) = map(_add_to_primal, x, t)
_add_to_primal(x::Tuple, t::Tuple) = map(_add_to_primal, x, t)
_add_to_primal(x::NamedTuple, t::NamedTuple) = map(_add_to_primal, x, t)
_add_to_primal(x, ::Tangent{NamedTuple{(), Tuple{}}}) = x

function _containerlike_add_to_primal(p::T, t::Union{Tangent, MutableTangent}) where {T}
    tmp = map(fieldnames(T)) do f
        tf = getfield(t.fields, f)
        isdefined(p, f) && is_init(tf) && return _add_to_primal(getfield(p, f), tf.tangent) 
        isdefined(p, f) && return getfield(p, f)
        throw(error("unable to handle undefined-ness"))
    end
    return T(tmp...)
end

"""
    _diff(p::P, q::P) where {P}

Required for testing.
_Not_ currently defined by default.
`_containerlike_diff` is potentially what you want to target when implementing for
a particular primal-tangent pair.

Computes the difference between `p` and `q`, which _must_ be of the same type, `P`.
Returns a tangent of type `tangent_type(P)`.
"""
function _diff(::P, ::P) where {P}
    tangent_type(P) === NoTangent && return NoTangent()
    T = Tangent{NamedTuple{(), Tuple{}}}
    tangent_type(P) === T && return T((;))
    error("tangent_type(P) is not NoTangent, and no other method provided")
end
_diff(p::P, q::P) where {P<:IEEEFloat} = p - q
_diff(p::P, q::P) where {P<:Array} = map(_diff, p, q)
_diff(p::P, q::P) where {P<:Union{Tuple, NamedTuple}} = map(_diff, p, q)

function _containerlike_diff(p::P, q::P) where {P}
    diffed_fields = map(fieldnames(P)) do f
        isdefined(p, f) && isdefined(q, f) && return _diff(getfield(p, f), getfield(q, f))
        throw(error("Unhandleable undefinedness"))
    end
    return build_tangent(P, diffed_fields...)
end

for _P in [
    UnitRange, Transpose, Adjoint, SubArray, Base.RefValue, LazyString, Diagonal, Xoshiro
]
    @eval _add_to_primal(p::$_P, t) = _containerlike_add_to_primal(p, t)
    @eval _diff(p::P, q::P) where {P<:$_P} = _containerlike_diff(p, q)
end
