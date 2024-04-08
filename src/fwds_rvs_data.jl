
"""
    NoFwdsData

Singleton type which indicates that there is nothing to be propagated on the forwards-pass
in addition to the primal data.
"""
struct NoFwdsData end

"""
    FwdsData(data::NamedTuple)

The component of a `struct` which is propagated alongside the primal on the forwards-pass of
AD. For example, the tangents for `Float64`s do not need to be propagated on the forwards-
pass of reverse-mode AD, so any `Float64` fields of `Tangent` do not need to appear in the
associated `FwdsData`.
"""
struct FwdsData{T<:NamedTuple}
    data::T
end

"""
    forwards_data_type(T)

Returns the type of the forwards data associated to a tangent of type `T`.
"""
forwards_data_type(T)

forwards_data_type(x) = throw(error("$x is not a type. Perhaps you meant typeof(x)?"))

forwards_data_type(::Type{T}) where {T<:IEEEFloat} = NoFwdsData

function forwards_data_type(::Type{PossiblyUninitTangent{T}}) where {T}
    Tfields = forwards_data_type(T)
    return PossiblyUninitTangent{Tfields}
end

function forwards_data_type(::Type{T}) where {T}

    # If the tangent type is NoTangent, then the forwards-component must be `NoFwdsData`.
    T == NoTangent && return NoFwdsData

    # This method can only handle struct types. Tell user to implement their own method.
    isprimitivetype(T) && throw(error(
        "$T is a primitive type. Implement a method of `forwards_data_type` for it."
    ))

    # If the type is a Union, then take the union type of its arguments.
    T isa Union && return Union{forwards_data_type(T.a), forwards_data_type(T.b)}

    # If the type is itself abstract, it's forward data could be anything.
    # The same goes for if the type has any undetermined type parameters.
    (isabstracttype(T) || !isconcretetype(T)) && return Any

    # If `P` is a mutable type, then its forwards data is its tangent.
    ismutabletype(T) && return T

    # If `P` is an immutable type, then some of its fields may not need to be propagated
    # on the forwards-pass.
    if T <: Tangent
        Tfields = fields_type(T)
        fwds_data_field_types = map(1:fieldcount(Tfields)) do n
            return forwards_data_type(fieldtype(Tfields, n))
        end
        all(==(NoFwdsData), fwds_data_field_types) && return NoFwdsData
        return FwdsData{NamedTuple{fieldnames(Tfields), Tuple{fwds_data_field_types...}}}
    end

    return :(error("Unhandled type $T"))
end

forwards_data_type(::Type{Ptr{P}}) where {P} = Ptr{tangent_type(P)}

function forwards_data_type(::Type{P}) where {P<:Tuple}
    isa(P, Union) && return Union{forwards_data_type(P.a), forwards_data_type(P.b)}
    isempty(P.parameters) && return NoFwdsData
    isa(last(P.parameters), Core.TypeofVararg) && return Any
    all(p -> forwards_data_type(p) == NoFwdsData, P.parameters) && return NoFwdsData
    return Tuple{map(forwards_data_type, fieldtypes(P))...}
end

function forwards_data_type(::Type{NamedTuple{names, T}}) where {names, T<:Tuple}
    if forwards_data_type(T) == NoFwdsData
        return NoFwdsData
    elseif isconcretetype(forwards_data_type(T))
        return NamedTuple{names, forwards_data_type(T)}
    else
        return Any
    end
end

"""
    forwards_data(t)::forwards_data_type(typeof(t))

Extract the forwards data from tangent `t`.
"""
function forwards_data(t::T) where {T}

    # Ask for the forwards-data type. Useful catch-all error checking for unexpected types.
    F = forwards_data_type(T)

    # Catch-all for anything with no forwards-data.
    F == NoFwdsData && return NoFwdsData()

    # Catch-all for anything where we return the whole object (mutable structs, arrays...).
    F == T && return t

    # T must be a `Tangent` by now. If it's not, something has gone wrong.
    !(T <: Tangent) && return :(error("Unhandled type $T"))
    return F(forwards_data(t.fields))
end

forwards_data(t::Tuple) = tuple_map(forwards_data, t)
forwards_data(t::NamedTuple{names}) where {names} = NamedTuple{names}(forwards_data(Tuple(t)))


"""
    NoRvsData()

Nothing to propagate backwards on the reverse-pass.
"""
struct NoRvsData end

"""
    RvsData(data::NamedTuple)

"""
struct RvsData{T<:NamedTuple}
    data::T
end

"""
    reverse_data_type(T)

Returns the type of the reverse data of a tangent of type T.
"""
reverse_data_type(T)

reverse_data_type(x) = throw(error("$x is not a type. Perhaps you meant typeof(x)?"))

reverse_data_type(::Type{T}) where {T<:IEEEFloat} = T

function reverse_data_type(::Type{PossiblyUninitTangent{T}}) where {T}
    return NoRvsData
    # return PossiblyUninitTangent{reverse_data_type(T)}
end

function reverse_data_type(::Type{T}) where {T}

    # If the tangent type is NoTangent, then the reverse-component must be `NoRvsData`.
    T == NoTangent && return NoRvsData

    # This method can only handle struct types. Tell user to implement their own method.
    isprimitivetype(T) && throw(error(
        "$T is a primitive type. Implement a method of `reverse_data_type` for it."
    ))

    # If the type is a Union, then take the union type of its arguments.
    T isa Union && return Union{reverse_data_type(T.a), reverse_data_type(T.b)}

    # If the type is itself abstract, it's reverse data could be anything.
    # The same goes for if the type has any undetermined type parameters.
    (isabstracttype(T) || !isconcretetype(T)) && return Any

    # If `P` is a mutable type, then all tangent info is propagated on the forwards-pass.
    ismutabletype(T) && return NoRvsData

    # If `T` is an immutable type, then some of its fields may not have been propagated on
    # the forwards-pass.
    if T <: Tangent
        Tfs = fields_type(T)
        rvs_types = map(n -> reverse_data_type(fieldtype(Tfs, n)), 1:fieldcount(Tfs))
        all(==(NoRvsData), rvs_types) && return NoRvsData
        return RvsData{NamedTuple{fieldnames(Tfs), Tuple{rvs_types...}}}
    end
end

reverse_data_type(::Type{<:Ptr}) = NoRvsData

function reverse_data_type(::Type{P}) where {P<:Tuple}
    isa(P, Union) && return Union{reverse_data_type(P.a), reverse_data_type(P.b)}
    isempty(P.parameters) && return NoRvsData
    isa(last(P.parameters), Core.TypeofVararg) && return Any
    all(p -> reverse_data_type(p) == NoRvsData, P.parameters) && return NoRvsData
    return Tuple{map(reverse_data_type, fieldtypes(P))...}
end

function reverse_data_type(::Type{NamedTuple{names, T}}) where {names, T<:Tuple}
    if reverse_data_type(T) == NoRvsData
        return NoRvsData
    elseif isconcretetype(reverse_data_type(T))
        return NamedTuple{names, reverse_data_type(T)}
    else
        return Any
    end
end

"""
    reverse_data(t)::reverse_data_type(typeof(t))

Extract the reverse data from tangent `t`.
"""
function reverse_data(t::T) where {T}

    # Ask for the reverse-data type. Useful catch-all error checking for unexpected types.
    R = reverse_data_type(T)

    # Catch-all for anything with no reverse-data.
    R == NoRvsData && return NoRvsData()

    # Catch-all for anything where we return the whole object (Float64, isbits structs, ...)
    R == T && return t

    # T must be a `Tangent` by now. If it's not, something has gone wrong.
    !(T <: Tangent) && return :(error("Unhandled type $T"))
    return reverse_data_type(T)(reverse_data(t.fields))
end

reverse_data(t::Tuple) = tuple_map(reverse_data, t)
reverse_data(t::NamedTuple{names}) where {names} = NamedTuple{names}(reverse_data(Tuple(t)))

"""
    combine_data(tangent_type, forwards_data, reverse_data)


"""
combine_data(::Type{NoTangent}, ::NoFwdsData, ::NoRvsData) = NoTangent()
combine_data(::Type{T}, ::NoFwdsData, r::T) where {T} = r
combine_data(::Type{F}, f::F, ::NoRvsData) where {F} = f
function combine_data(::Type{T}, f::Tuple, r::Tuple) where {T}
    return T(map(combine_data, fieldtypes(T), f, r))
end
function combine_data(::Type{T}, f::NamedTuple{ns}, r::NamedTuple{ns}) where {T, ns}
    t = map(combine_data, fieldtypes(T), Tuple(f), Tuple(r))
    return T(NamedTuple{ns}(t))
end
function combine_data(::Type{T}, f::FwdsData{F}, r::RvsData{R}) where {T<:Tangent, F, R}
    nt = fields_type(T)
    Ts = fieldtypes(nt)
    return T(nt(tuple(map(combine_data, Ts, f.data, r.data)...)))
end

combine_data(::Type{T}, f::FwdsData{F}, ::NoRvsData) where {T<:Tangent, F} = T(f.data)

"""
    zero_tangent(p, ::NoFwdsData)


"""
zero_tangent(p, ::NoFwdsData) = zero_tangent(p)

function zero_tangent(p::P, f::F) where {P, F}
    T = tangent_type(P)
    T == F && return f
    r = reverse_data(zero_tangent(p))
    return combine_data(tangent_type(P), f, r)
end

zero_tangent(p::Tuple, f::Tuple) = tuple_map(zero_tangent, p, f)
function zero_tangent(p::NamedTuple{names}, f::NamedTuple{names}) where {names}
    return NamedTuple{names}(zero_tangent(Tuple(p), Tuple(f)))
end
