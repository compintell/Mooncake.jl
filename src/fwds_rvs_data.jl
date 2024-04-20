
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
    ZeroRData()

Singleton type indicating zero-valued rdata. This should only ever appear as an
intermediate quantity in the reverse-pass of AD when the type of the primal is not fully
inferable, or a field of a type is abstractly typed.

If you see this anywhere in actual code, or if it appears in a hand-written rule, this is an
error -- please open an issue in such a situation.
"""
struct ZeroRData end

@inline increment!!(::ZeroRData, r::R) where {R} = r

"""

"""
zero_rdata_from_type(::Type{P}) where {P}

zero_rdata_from_type(::Type{P}) where {P<:IEEEFloat} = zero(P)

@generated function zero_rdata_from_type(::Type{P}) where {P}

    # Get types associated to primal.
    T = tangent_type(P)
    R = rdata_type(T)

    # If there's no reverse data, return no reverse data, e.g. for mutable types.
    R == NoRData && return NoRData()

    # If the type is itself abstract, it's reverse data could be anything.
    # The same goes for if the type has any undetermined type parameters.
    (isabstracttype(T) || !isconcretetype(T)) && return ZeroRData()

    # T ought to be a `Tangent`. If it's not, something has gone wrong.
    !(T <: Tangent) && return Expr(:call, error, "Unhandled type $T")
    rdata_field_zeros_exprs = ntuple(fieldcount(P)) do n
        R_field = rdata_field_type(P, n)
        if R_field <: PossiblyUninitTangent
            return :($R_field(zero_rdata_from_type($(fieldtype(P, n)))))
        else
            return :(zero_rdata_from_type($(fieldtype(P, n))))
        end
    end
    backing_data_expr = Expr(:call, :tuple, rdata_field_zeros_exprs...)
    backing_expr = :($(rdata_backing_type(P))($backing_data_expr))
    return Expr(:call, R, backing_expr)
end

@generated function zero_rdata_from_type(::Type{P}) where {P<:Tuple}
    # Get types associated to primal.
    T = tangent_type(P)
    R = rdata_type(T)

    # If there's no reverse data, return no reverse data, e.g. for mutable types.
    R == NoRData && return NoRData()

    return Expr(:call, tuple, map(p -> :(zero_rdata_from_type($p)), P.parameters)...)
end

@generated function zero_rdata_from_type(::Type{NamedTuple{names, Pt}}) where {names, Pt}

    # Get types associated to primal.
    P = NamedTuple{names, Pt}
    T = tangent_type(P)
    R = rdata_type(T)

    # If there's no reverse data, return no reverse data, e.g. for mutable types.
    R == NoRData && return NoRData()

    return :(NamedTuple{$names}(zero_rdata_from_type($Pt)))
end

"""
    LazyZeroRData{P, Tdata}()

This type is a lazy placeholder for `zero_rdata_from_type`. This is used to defer
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

# Attempt to make the construction of the zero rdata element as lazy as possible. Fallback
# to not being lazy if we cannot prove it's safe to defer construction -- just store the
# entire object.
@inline function LazyZeroRData(p::P) where {P}
    R = rdata_type(tangent_type(P))

    R == NoRData && return LazyZeroRData{P, Nothing}(nothing)

    return _lazy_zero_rdata_ctor_fallback(p)
end

# Fallback just constructs the rdata, and stores it.
@inline function _lazy_zero_rdata_ctor_fallback(p::P) where {P}
    rdata = zero_rdata(p)
    return LazyZeroRData{P, _typeof(rdata)}(rdata)
end

@inline function instantiate(r::LazyZeroRData{P}) where {P}
    R = rdata_type(tangent_type(P))

    R == NoRData && return zero_rdata_from_type(P)

    return r.data
end

# For many important types, no data is required to be stored in order to construct an
# instance of the zero rdata for `P`. All of the code below here is designed to make use of
# that to avoid intermediate store when it's not necessary.

@inline LazyZeroRData(::P) where {P<:IEEEFloat} = LazyZeroRData{P, Nothing}(nothing)
@inline instantiate(::LazyZeroRData{P}) where {P<:IEEEFloat} = zero_rdata_from_type(P)

"""
    combine_data(tangent_type, fdata, rdata)


"""
combine_data(::Type{NoTangent}, ::NoFData, ::NoRData) = NoTangent()
combine_data(::Type{T}, ::NoFData, r::T) where {T} = r
combine_data(::Type{T}, f::T, ::NoRData) where {T} = f

# General structs.
function combine_data(::Type{T}, f::FData{F}, r::RData{R}) where {T<:Tangent, F, R}
    return __combine_struct_data(T, f.data, r.data)
end

function combine_data(::Type{T}, f::NoFData, r::RData{R}) where {T<:Tangent, R}
    return __combine_struct_data(T, tuple_map(_ -> NoFData(), r.data), r.data)
end

function combine_data(::Type{T}, f::FData{F}, ::NoRData) where {T<:Tangent, F}
    return __combine_struct_data(T, f.data, tuple_map(_ -> NoRData(), f.data))
end

# Helper used for combining fdata and rdata for (mutable) structs.
function __combine_struct_data(::Type{T}, fs, rs) where {T<:Tangent}
    nt = fields_type(T)
    return T(nt(map(combine_data, fieldtypes(nt), fs, rs)))
end

function combine_data(
    ::Type{PossiblyUninitTangent{T}}, f::PossiblyUninitTangent, r::PossiblyUninitTangent
) where {T}
    Tout = PossiblyUninitTangent{T}
    return is_init(f) ? Tout(combine_data(T, val(f), val(r))) : Tout()
end

# Tuples and NamedTuples
function combine_data(::Type{T}, f::Tuple, r::Tuple) where {T}
    return T(map(combine_data, fieldtypes(T), f, r))
end
function combine_data(::Type{T}, f::NamedTuple{ns}, r::NamedTuple{ns}) where {T, ns}
    return T(map(combine_data, fieldtypes(T), f, r))
end
function combine_data(::Type{T}, f::Union{Tuple, NamedTuple}, r::NoRData) where {T}
    return combine_data(T, f, map(_ -> NoRData(), f))
end
function combine_data(::Type{T}, f::NoFData, r::Union{Tuple, NamedTuple}) where {T}
    return combine_data(T, map(_ -> NoFData(), r), r)
end

"""
    zero_tangent(p, ::NoFData)


"""
zero_tangent(p, ::NoFData) = zero_tangent(p)

function zero_tangent(p::P, f::F) where {P, F}
    T = tangent_type(P)
    T == F && return f
    r = rdata(zero_tangent(p))
    return combine_data(tangent_type(P), f, r)
end

zero_tangent(p::Tuple, f::Union{Tuple, NamedTuple}) = tuple_map(zero_tangent, p, f)
