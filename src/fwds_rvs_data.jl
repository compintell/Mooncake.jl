"""
    NoFData

Singleton type which indicates that there is nothing to be propagated on the forwards-pass
in addition to the primal data.
"""
struct NoFData end

increment!!(::NoFData, ::NoFData) = NoFData()

"""
    FData(data::NamedTuple)

The component of a `struct` which is propagated alongside the primal on the forwards-pass of
AD. For example, the tangents for `Float64`s do not need to be propagated on the forwards-
pass of reverse-mode AD, so any `Float64` fields of `Tangent` do not need to appear in the
associated `FData`.
"""
struct FData{T<:NamedTuple}
    data::T
end

fields_type(::Type{FData{T}}) where {T<:NamedTuple} = T

increment!!(x::F, y::F) where {F<:FData} = F(tuple_map(increment!!, x.data, y.data))

"""
    fdata_type(T)

Returns the type of the forwards data associated to a tangent of type `T`.
"""
fdata_type(T)

fdata_type(x) = throw(error("$x is not a type. Perhaps you meant typeof(x)?"))

fdata_type(::Type{T}) where {T<:IEEEFloat} = NoFData

function fdata_type(::Type{PossiblyUninitTangent{T}}) where {T}
    Tfields = fdata_type(T)
    return PossiblyUninitTangent{Tfields}
end

@generated function fdata_type(::Type{T}) where {T}

    # If the tangent type is NoTangent, then the forwards-component must be `NoFData`.
    T == NoTangent && return NoFData

    # This method can only handle struct types. Tell user to implement their own method.
    isprimitivetype(T) && throw(error(
        "$T is a primitive type. Implement a method of `fdata_type` for it."
    ))

    # If the type is a Union, then take the union type of its arguments.
    T isa Union && return Union{fdata_type(T.a), fdata_type(T.b)}

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
            return fdata_type(fieldtype(Tfields, n))
        end
        all(==(NoFData), fwds_data_field_types) && return NoFData
        return FData{NamedTuple{fieldnames(Tfields), Tuple{fwds_data_field_types...}}}
    end

    return :(error("Unhandled type $T"))
end

fdata_type(::Type{Ptr{P}}) where {P} = Ptr{tangent_type(P)}

@generated function fdata_type(::Type{P}) where {P<:Tuple}
    isa(P, Union) && return Union{fdata_type(P.a), fdata_type(P.b)}
    isempty(P.parameters) && return NoFData
    isa(last(P.parameters), Core.TypeofVararg) && return Any
    all(p -> fdata_type(p) == NoFData, P.parameters) && return NoFData
    return Tuple{map(fdata_type, fieldtypes(P))...}
end

@generated function fdata_type(::Type{NamedTuple{names, T}}) where {names, T<:Tuple}
    if fdata_type(T) == NoFData
        return NoFData
    elseif isconcretetype(fdata_type(T))
        return NamedTuple{names, fdata_type(T)}
    else
        return Any
    end
end

"""
    fdata_field_type(::Type{P}, n::Int) where {P}

Returns the type of to the nth field of the fdata type associated to `P`. Will be a
`PossiblyUninitTangent` if said field can be undefined.
"""
function fdata_field_type(::Type{P}, n::Int) where {P}
    Tf = tangent_type(fieldtype(P, n))
    f = ismutabletype(P) ? Tf : fdata_type(Tf)
    return is_always_initialised(P, n) ? f : _wrap_type(f)
end

"""
    fdata(t)::fdata_type(typeof(t))

Extract the forwards data from tangent `t`.
"""
@generated function fdata(t::T) where {T}

    # Ask for the forwards-data type. Useful catch-all error checking for unexpected types.
    F = fdata_type(T)

    # Catch-all for anything with no forwards-data.
    F == NoFData && return :(NoFData())

    # Catch-all for anything where we return the whole object (mutable structs, arrays...).
    F == T && return :(t)

    # T must be a `Tangent` by now. If it's not, something has gone wrong.
    !(T <: Tangent) && return :(error("Unhandled type $T"))
    return :($F(fdata(t.fields)))
end

function fdata(t::T) where {T<:PossiblyUninitTangent}
    F = fdata_type(T)
    return is_init(t) ? F(fdata(val(t))) : F()
end

@generated function fdata(t::Union{Tuple, NamedTuple})
    fdata_type(t) == NoFData && return NoFData()
    return :(tuple_map(fdata, t))
end

uninit_fdata(p) = fdata(uninit_tangent(p))

"""
    NoRData()

Nothing to propagate backwards on the reverse-pass.
"""
struct NoRData end

@inline increment!!(::NoRData, ::NoRData) = NoRData()

@inline increment_field!!(::NoRData, y, ::Val) = NoRData()

"""
    RData(data::NamedTuple)

"""
struct RData{T<:NamedTuple}
    data::T
end

fields_type(::Type{RData{T}}) where {T<:NamedTuple} = T

@inline increment!!(x::RData{T}, y::RData{T}) where {T} = RData(increment!!(x.data, y.data))

@inline function increment_field!!(x::RData{T}, y, ::Val{f}) where {T, f}
    y isa NoRData && return x
    new_val = fieldtype(T, f) <: PossiblyUninitTangent ? fieldtype(T, f)(y) : y
    return RData(increment_field!!(x.data, new_val, Val(f)))
end

"""
    rdata_type(T)

Returns the type of the reverse data of a tangent of type T.
"""
rdata_type(T)

rdata_type(x) = throw(error("$x is not a type. Perhaps you meant typeof(x)?"))

rdata_type(::Type{T}) where {T<:IEEEFloat} = T

function rdata_type(::Type{PossiblyUninitTangent{T}}) where {T}
    return PossiblyUninitTangent{rdata_type(T)}
end

@generated function rdata_type(::Type{T}) where {T}

    # If the tangent type is NoTangent, then the reverse-component must be `NoRData`.
    T == NoTangent && return NoRData

    # This method can only handle struct types. Tell user to implement their own method.
    isprimitivetype(T) && throw(error(
        "$T is a primitive type. Implement a method of `rdata_type` for it."
    ))

    # If the type is a Union, then take the union type of its arguments.
    T isa Union && return Union{rdata_type(T.a), rdata_type(T.b)}

    # If the type is itself abstract, it's reverse data could be anything.
    # The same goes for if the type has any undetermined type parameters.
    (isabstracttype(T) || !isconcretetype(T)) && return Any

    # If `P` is a mutable type, then all tangent info is propagated on the forwards-pass.
    ismutabletype(T) && return NoRData

    # If `T` is an immutable type, then some of its fields may not have been propagated on
    # the forwards-pass.
    if T <: Tangent
        Tfs = fields_type(T)
        rvs_types = map(n -> rdata_type(fieldtype(Tfs, n)), 1:fieldcount(Tfs))
        all(==(NoRData), rvs_types) && return NoRData
        return RData{NamedTuple{fieldnames(Tfs), Tuple{rvs_types...}}}
    end
end

rdata_type(::Type{<:Ptr}) = NoRData

@generated function rdata_type(::Type{P}) where {P<:Tuple}
    isa(P, Union) && return Union{rdata_type(P.a), rdata_type(P.b)}
    isempty(P.parameters) && return NoRData
    isa(last(P.parameters), Core.TypeofVararg) && return Any
    all(p -> rdata_type(p) == NoRData, P.parameters) && return NoRData
    return Tuple{map(rdata_type, fieldtypes(P))...}
end

function rdata_type(::Type{NamedTuple{names, T}}) where {names, T<:Tuple}
    if rdata_type(T) == NoRData
        return NoRData
    elseif isconcretetype(rdata_type(T))
        return NamedTuple{names, rdata_type(T)}
    else
        return Any
    end
end

"""
    rdata_field_type(::Type{P}, n::Int) where {P}

Returns the type of to the nth field of the rdata type associated to `P`. Will be a
`PossiblyUninitTangent` if said field can be undefined.
"""
function rdata_field_type(::Type{P}, n::Int) where {P}
    r = rdata_type(tangent_type(fieldtype(P, n)))
    return is_always_initialised(P, n) ? r : _wrap_type(r)
end

"""
    rdata(t)::rdata_type(typeof(t))

Extract the reverse data from tangent `t`.
"""
@generated function rdata(t::T) where {T}

    # Ask for the reverse-data type. Useful catch-all error checking for unexpected types.
    R = rdata_type(T)

    # Catch-all for anything with no reverse-data.
    R == NoRData && return :(NoRData())

    # Catch-all for anything where we return the whole object (Float64, isbits structs, ...)
    R == T && return :(t)

    # T must be a `Tangent` by now. If it's not, something has gone wrong.
    !(T <: Tangent) && return :(error("Unhandled type $T"))
    return :($(rdata_type(T))(rdata(t.fields)))
end

function rdata(t::T) where {T<:PossiblyUninitTangent}
    R = rdata_type(T)
    return is_init(t) ? R(rdata(val(t))) : R()
end

@generated function rdata(t::Union{Tuple, NamedTuple})
    rdata_type(t) == NoRData && return NoRData()
    return :(tuple_map(rdata, t))
end

function rdata_backing_type(::Type{P}) where {P}
    rdata_field_types = map(n -> rdata_field_type(P, n), 1:fieldcount(P))
    all(==(NoRData), rdata_field_types) && return NoRData
    return NamedTuple{fieldnames(P), Tuple{rdata_field_types...}}
end

"""
    zero_rdata(p)

Given value `p`, return the zero element associated to its reverse data type.
"""
zero_rdata(p)

zero_rdata(p::IEEEFloat) = zero(p)

@generated function zero_rdata(p::P) where {P}

    # Get types associated to primal.
    T = tangent_type(P)
    R = rdata_type(T)

    # If there's no reverse data, return no reverse data, e.g. for mutable types.
    R == NoRData && return :(NoRData())

    # T ought to be a `Tangent`. If it's not, something has gone wrong.
    !(T <: Tangent) && Expr(:call, error, "Unhandled type $T")
    rdata_field_zeros_exprs = ntuple(fieldcount(P)) do n
        R_field = rdata_field_type(P, n)
        if R_field <: PossiblyUninitTangent
            return :(isdefined(p, $n) ? $R_field(zero_rdata(getfield(p, $n))) : $R_field())
        else
            return :(zero_rdata(getfield(p, $n)))
        end
    end
    backing_data_expr = Expr(:call, :tuple, rdata_field_zeros_exprs...)
    backing_expr = :($(rdata_backing_type(P))($backing_data_expr))
    return Expr(:call, R, backing_expr)
end

@generated function zero_rdata(p::Union{Tuple, NamedTuple})
    rdata_type(tangent_type(p)) == NoRData && return NoRData()
    return :(tuple_map(zero_rdata, p))
end

"""
    can_produce_zero_rdata_from_type(::Type{P}) where {P}

Returns whether or not the zero element of the rdata type for primal type `P` can be
obtained from `P` alone.
"""
@generated function can_produce_zero_rdata_from_type(::Type{P}) where {P}
    R = rdata_type(tangent_type(P))
    R == NoRData && return true
    return isstructtype(P) ? all(can_produce_zero_rdata_from_type, fieldtypes(P)) : false
end

can_produce_zero_rdata_from_type(::Type{<:IEEEFloat}) = true

"""
    CannotProduceZeroRDataFromType()

Returned by `zero_rdata_from_type` if is not possible to construct the zero rdata element
for a given type. See `zero_rdata_from_type` for more info.
"""
struct CannotProduceZeroRDataFromType end

"""
    zero_rdata_from_type(::Type{P}) where {P}

Returns the zero element of `rdata_type(tangent_type(P))` if this is possible given only
`P`. If not possible, returns an instance of `CannotProduceZeroRDataFromType`.

For example, the zero rdata associated to any primal of type `Float64` is `0.0`, so for
`Float64`s this function is simple. Similarly, if the rdata type for `P` is `NoRData`, that
can simply be returned.

However, it is not possible to return the zero rdata element for abstract types e.g. `Real`
as the type does not uniquely determine the zero element -- the rdata type for `Real` is
`Any`.

These considerations apply recursively to tuples / namedtuples / structs, etc.

If you encounter a type which this function returns `CannotProduceZeroRDataFromType`, but
you believe this is done in error, please open an issue. This kind of problem does not
constitute a correctness problem, but can be detrimental to performance, so should be dealt
with.
"""
function zero_rdata_from_type(::Type{P}) where {P}
    R = rdata_type(tangent_type(P))

    # Simple case.
    R == NoRData && return NoRData()

    # If `P` is a struct type, attempt to derive the zero rdata for it. We cannot derive
    # the zero rdata if it is not possible to derive the zero rdata for any of its fields.
    if isstructtype(P)
        names = fieldnames(P)
        field_zeros = tuple_map(zero_rdata_from_type, fieldtypes(P))
        if all(tuple_map(z -> !(z isa CannotProduceZeroRDataFromType), field_zeros))
            wrapped_field_zeros = tuple_map(ntuple(identity, length(names))) do n
                fzero = field_zeros[n]
                if tangent_field_type(P, n) <: PossiblyUninitTangent
                    return _wrap_field(rdata_type(tangent_type(fieldtype(P, n))), fzero)
                else
                    return fzero
                end
            end
            return R(NamedTuple{names}(wrapped_field_zeros))
        end
    end

    # Fallback -- we've not been able to figure out how to produce an instance of zero rdata
    # so report that it cannot be done.
    return CannotProduceZeroRDataFromType()
end

function zero_rdata_from_type(::Type{P}) where {P<:Tuple}
    R = rdata_type(tangent_type(P))

    R == NoRData && return NoRData()

    field_zeros = tuple_map(zero_rdata_from_type, fieldtypes(P))
    if all(tuple_map(z -> !(z isa CannotProduceZeroRDataFromType), field_zeros))
        return field_zeros
    end

    return CannotProduceZeroRDataFromType()
end

function zero_rdata_from_type(::Type{P}) where {P<:NamedTuple}
    R = rdata_type(tangent_type(P))

    R == NoRData && return NoRData()

    field_zeros = tuple_map(zero_rdata_from_type, fieldtypes(P))
    if all(tuple_map(z -> !(z isa CannotProduceZeroRDataFromType), field_zeros))
        return NamedTuple{fieldnames(P)}(field_zeros)
    end

    return CannotProduceZeroRDataFromType()
end

zero_rdata_from_type(::Type{P}) where {P<:IEEEFloat} = zero(P)

"""
    LazyZeroRData{P, Tdata}()

This type is a lazy placeholder for `zero_like_rdata_from_type`. This is used to defer
construction of zero data to the reverse pass. Calling `instantiate` on an instance of this
will construct a zero data.

Users should construct using `LazyZeroRData(p)`, where `p` is an value of type `P`. This
constructor, and `instantiate`, are specialised to minimise the amount of data which must
be stored. For example, `Float64`s do not need any data, so `LazyZeroRData(0.0)` produces
an instance of a singleton type, meaning that various important optimisations can be
performed in AD.
"""
struct LazyZeroRData{P, Tdata}
    data::Tdata
end

# Be lazy if we can compute the zero element given only the type, otherwise just store the
# zero element and use it later.
@inline function LazyZeroRData(p::P) where {P}
    if zero_rdata_from_type(P) isa CannotProduceZeroRDataFromType
        rdata = zero_rdata(p)
        return LazyZeroRData{P, _typeof(rdata)}(rdata)
    else
        return LazyZeroRData{P, Nothing}(nothing)
    end
end

@inline instantiate(::LazyZeroRData{P, Nothing}) where {P} = zero_rdata_from_type(P)
@inline instantiate(r::LazyZeroRData) = r.data

"""
    tangent_type(F::Type, R::Type)::Type

Given the type of the fdata and rdata, `F` and `R` resp., for some primal type, compute its
tangent type. This method must be equivalent to `tangent_type(_typeof(primal))`.
"""
tangent_type(::Type{NoFData}, ::Type{NoRData}) = NoTangent
tangent_type(::Type{NoFData}, ::Type{R}) where {R<:IEEEFloat} = R
tangent_type(::Type{F}, ::Type{NoRData}) where {F<:Array} = F

# Tuples
function tangent_type(::Type{F}, ::Type{R}) where {F<:Tuple, R<:Tuple}
    return Tuple{tuple_map(tangent_type, Tuple(F.parameters), Tuple(R.parameters))...}
end
function tangent_type(::Type{NoFData}, ::Type{R}) where {R<:Tuple}
    F_tuple = Tuple{tuple_fill(NoFData, Val(length(R.parameters)))...}
    return tangent_type(F_tuple, R)
end
function tangent_type(::Type{F}, ::Type{NoRData}) where {F<:Tuple}
    R_tuple = Tuple{tuple_fill(NoRData, Val(length(F.parameters)))...}
    return tangent_type(F, R_tuple)
end

# NamedTuples
function tangent_type(::Type{F}, ::Type{R}) where {ns, F<:NamedTuple{ns}, R<:NamedTuple{ns}}
    return NamedTuple{ns, tangent_type(tuple_type(F), tuple_type(R))}
end
function tangent_type(::Type{NoFData}, ::Type{R}) where {ns, R<:NamedTuple{ns}}
    return NamedTuple{ns, tangent_type(NoFData, tuple_type(R))}
end
function tangent_type(::Type{F}, ::Type{NoRData}) where {ns, F<:NamedTuple{ns}}
    return NamedTuple{ns, tangent_type(tuple_type(F), NoRData)}
end
tuple_type(::Type{<:NamedTuple{<:Any, T}}) where {T<:Tuple} = T

# mutable structs
tangent_type(::Type{F}, ::Type{NoRData}) where {F<:MutableTangent} = F

# structs
function tangent_type(::Type{F}, ::Type{R}) where {F<:FData, R<:RData}
    return Tangent{tangent_type(fields_type(F), fields_type(R))}
end
function tangent_type(::Type{NoFData}, ::Type{R}) where {R<:RData}
    return Tangent{tangent_type(NoFData, fields_type(R))}
end
function tangent_type(::Type{F}, ::Type{NoRData}) where {F<:FData}
    return Tangent{tangent_type(fields_type(F), NoRData)}
end

function tangent_type(
    ::Type{PossiblyUninitTangent{F}}, ::Type{PossiblyUninitTangent{R}}
) where {F, R}
    return PossiblyUninitTangent{tangent_type(F, R)}
end

# Abstract types.
tangent_type(::Type{Any}, ::Type{Any}) = Any


"""
    tangent(f, r)

Reconstruct the tangent `t` for which `fdata(t) == f` and `rdata(t) == r`.
"""
tangent(::NoFData, ::NoRData) = NoTangent()
tangent(::NoFData, r::IEEEFloat) = r
tangent(f::Array, ::NoRData) = f

# Tuples
tangent(f::Tuple, r::Tuple) = tuple_map(tangent, f, r)
tangent(::NoFData, r::Tuple) = tuple_map(_r -> tangent(NoFData(), _r), r)
tangent(f::Tuple, ::NoRData) = tuple_map(_f -> tangent(_f, NoRData()), f)

# NamedTuples
function tangent(f::NamedTuple{n}, r::NamedTuple{n}) where {n}
    return NamedTuple{n}(tangent(Tuple(f), Tuple(r)))
end
function tangent(::NoFData, r::NamedTuple{ns}) where {ns}
    return NamedTuple{ns}(tangent(NoFData(), Tuple(r)))
end
function tangent(f::NamedTuple{ns}, ::NoRData) where {ns}
    return NamedTuple{ns}(tangent(Tuple(f), NoRData()))
end

# mutable structs
tangent(f::MutableTangent, r::NoRData) = f

# structs
function tangent(f::F, r::R) where {F<:FData, R<:RData}
    return tangent_type(F, R)(tangent(f.data, r.data))
end
function tangent(::NoFData, r::R) where {R<:RData}
    return tangent_type(NoFData, R)(tangent(NoFData(), r.data))
end
function tangent(f::F, ::NoRData) where {F<:FData}
    return tangent_type(F, NoRData)(tangent(f.data, NoRData()))
end

function tangent(f::PossiblyUninitTangent{F}, r::PossiblyUninitTangent{R}) where {F, R}
    T = PossiblyUninitTangent{tangent_type(F, R)}
    is_init(f) && is_init(r) && return T(tangent(val(f), val(r)))
    !is_init(f) && !is_init(r) && return T()
    throw(ArgumentError("Initialisation mismatch"))
end
function tangent(f::PossiblyUninitTangent{F}, ::PossiblyUninitTangent{NoRData}) where {F}
    T = PossiblyUninitTangent{tangent_type(F, NoRData)}
    return is_init(f) ? T(tangent(val(f), NoRData())) : T()
end
function tangent(::PossiblyUninitTangent{NoFData}, r::PossiblyUninitTangent{R}) where {R}
    T = PossiblyUninitTangent{tangent_type(NoFData, R)}
    return is_init(r) ? T(tangent(NoFData(), val(r))) : T()
end
function tangent(::PossiblyUninitTangent{NoFData}, ::PossiblyUninitTangent{NoRData})
    return PossiblyUninitTangent(NoTangent())
end

"""
    zero_tangent(p, ::NoFData)


"""
zero_tangent(p, ::NoFData) = zero_tangent(p)

function zero_tangent(p::P, f::F) where {P, F}
    T = tangent_type(P)
    T == F && return f
    r = rdata(zero_tangent(p))
    return tangent(f, r)
end

zero_tangent(p::Tuple, f::Union{Tuple, NamedTuple}) = tuple_map(zero_tangent, p, f)
