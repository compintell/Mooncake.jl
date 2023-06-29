# IGNORE THIS COMMENT. NEEDS REVISING!
# The entire set of valid operations on tangents is:
# - ==(::AbstractTangent, ::AbstractTangent). Must not error.
# - +(::AbstractTangent, ::AbstractTangent). Must not error.
# - *(a::Float64, ::AbstractTangent). Must not error.
#
# The entire set of functions that we add to primals / their types is:
# - tangent_type(::Type{P}) = tangent-type of primal type. Must not error.
# - zero_tangent(x) = zero tangent associated to primal x. Must not error.
# - randn_tangent(rng, x) = same as zero_tangent, but each dimension sampled from zero-mean
#   Must not error.
# - increment!!(x::P, ẋ::T)::P where P is a primal type, and T must be the tangent
#   type output by `tangent_type(T)`. Adds `ẋ` to `x` if possible. May throw an error if
#   `x + ẋ` is not a valid location at which to construct an instance of type `P`.
#
# Currently, this code focusses on correctness / coverage, rather than performance.
# Will need to be optimised via generated functions in reality. This is definitely doable
# though.

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

_wrap_type(::Type{NoTangent}) = NoTangent
_wrap_type(::Type{T}) where {T} = PossiblyUninitTangent{T}

_wrap_field(::Type{Q}, x::NoTangent) where {Q} = x
_wrap_field(::Type{Q}, x::T) where {Q, T} = PossiblyUninitTangent{Q}(x)

_wrap_field(x::T) where {T} = _wrap_field(T, x)

_wrap_zero_field(::Type{NoTangent}) = NoTangent()
_wrap_zero_field(::Type{Q}) where {Q} = PossiblyUninitTangent(Q)

struct Tangent{Tfields<:NamedTuple}
    fields::Tfields
end

Base.:(==)(x::Tangent, y::Tangent) = x.fields == y.fields

mutable struct MutableTangent{Tfields<:NamedTuple}
    fields::Tfields
end

Base.:(==)(x::MutableTangent, y::MutableTangent) = x.fields == y.fields

"""
    tangent_type(T)

The type used to represent the tangent of an object of type `T` must be unique, and
determined entirely by its type.

If this function runs on a given type, then its possible to verify that all other functions
produce the correct type. It's also straightforward to ensure that all of the types in a
given module have a `tangent_type` defined using this function and a simple testing utility.
This is important because it's necessary for every single type in the type system to have
its `tangent_type` defined.
Fortunately, these can mostly be derived automatically.

This restriction exists to ensure that it's possible to achieve type-stability in dynamic
code. It prevents some kinds of optimisation from being possible. Notably, those derived
from the use of `ZeroTangent`s and `Thunk`s.
"""
tangent_type(T)

function tangent_type(x)
    throw(error("$x is not a type. Perhaps you meant typeof(x)?"))
end

tangent_type(::Type{<:DataType}) = NoTangent

tangent_type(::Type{Ptr{P}}) where {P} = Ptr{tangent_type(P)}

tangent_type(::Type{<:Ptr}) = NoTangent

tangent_type(::Type{Bool}) = NoTangent

tangent_type(::Type{Char}) = NoTangent

tangent_type(::Type{Union{}}) = NoTangent

tangent_type(::Type{P}) where {P<:Union{UInt8, UInt16, UInt32, UInt64, UInt128}} = NoTangent

tangent_type(::Type{P}) where {P<:Union{Int8, Int16, Int32, Int64, Int128}} = NoTangent

tangent_type(::Type{P}) where {P<:IEEEFloat} = P

tangent_type(::Type{<:Core.LLVMPtr}) = NoTangent

function tangent_type(::Type{<:Array{P, N}}) where {P, N}
    return tangent_type(P) === NoTangent ? NoTangent : Array{tangent_type(P), N}
end

tangent_type(::Type{<:Array{P, N} where {P}}) where {N} = Array{Any, N}

tangent_type(::Type{<:MersenneTwister}) = NoTangent

tangent_type(u::Type{Union}) = Union{tangent_type(u.a), tangent_type(u.b)}

tangent_type(::Type{T}) where {T<:Tuple} = Tuple{map(tangent_type, fieldtypes(T))...}

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

    # If P has no fields / is a singleton-type, it must have no tangent.
    Base.issingletontype(P) && return NoTangent

    # Derive tangent type.
    tangent_field_names = fieldnames(P)
    tangent_field_types = map(t -> _wrap_type(tangent_type(t)), fieldtypes(P))
    if all(tangent_field_types .== NoTangent)
        return NoTangent
    else
        T = ismutabletype(P) ? MutableTangent : Tangent
        return T{NamedTuple{tangent_field_names, Tuple{tangent_field_types...}}}
    end
end

"""
    randn_tangent(rng::AbstractRNG, x::T) where {T}

Generate a randomly-chosen tangent to `x`.
"""
randn_tangent(::AbstractRNG, ::NoTangent) = NoTangent()
randn_tangent(rng::AbstractRNG, ::T) where {T<:IEEEFloat} = randn(rng, T)
function randn_tangent(rng::AbstractRNG, x::Array{T}) where {T}
    tangent_type(T) === NoTangent && return NoTangent()
    return map(x -> randn_tangent(rng, x), x)
end
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
        if !isdefined(x, field_name)
            return _wrap_zero_field(T)
        else
            return _wrap_field(T, randn_tangent(rng, getfield(x, field_name)))
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
increment!!(x::T, y::T) where {T<:Array} = x .+= y
increment!!(x::T, y::T) where {T<:Tuple} = map(increment!!, x, y)
increment!!(x::T, y::T) where {T<:NamedTuple} = map(increment!!, x, y)
function increment!!(x::T, y::T) where {T<:PossiblyUninitTangent}
    is_init(x) && is_init(y) && return __PUT(increment!!(x.tangent, y.tangent))
    !is_init(x) && !is_init(y) && return x
    is_init(x) && !is_init(y) && error("x is initialised, but y is not")
    !is_init(x) && is_init(y) && error("x is not initialised, but y is")
end
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
    set_to_zero!!(x)

Set `x` to its zero element (`x` should be a tangent, so the zero must exist).
"""
set_to_zero!!(::NoTangent) = NoTangent()
set_to_zero!!(x::Base.IEEEFloat) = zero(x)
set_to_zero!!(x::Union{Tuple, NamedTuple}) = map(set_to_zero!!, x)
set_to_zero!!(x::Array) = map!(set_to_zero!!, x, x)
function set_to_zero!!(x::PossiblyUninitTangent)
    return is_init(x) ? __PUT(set_to_zero!!(x.tangent)) : x
end
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

"""
    zero_tangent(x)

Returns the unique zero element of the tangent space of `x`.
It is an error for the zero element of the tangent space of `x` to be represented by
anything other than that which this function returns.
"""
zero_tangent(x)
zero_tangent(::Union{Int8, Int16, Int32, Int64, Int128}) = NoTangent()
zero_tangent(x::IEEEFloat) = zero(x)
function zero_tangent(x::Array{P, N}) where {P, N}
    return tangent_type(P) == NoTangent ? NoTangent() : map(zero_tangent, x)
end
zero_tangent(x::Union{Tuple, NamedTuple}) = map(zero_tangent, x)
function zero_tangent(x::P) where {P}

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
        if !isdefined(x, field_name)
            return _wrap_zero_field(T)
        else
            return _wrap_field(T, zero_tangent(getfield(x, field_name)))
        end
    end
    if all(tangent_field_zeros .== NoTangent())
        return NoTangent()
    else
        wrapped_tangent_field_types = map(_wrap_type, tangent_field_types)
        Tbacking = NamedTuple{tangent_field_names, Tuple{wrapped_tangent_field_types...}}
        backing = Tbacking(tangent_field_zeros)
        T = ismutabletype(P) ? MutableTangent : Tangent
        return T{Tbacking}(backing)
    end
end
