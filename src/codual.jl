struct CoDual{Tx,Tdx}
    x::Tx
    dx::Tdx
end

# Always sharpen the first thing if it's a type so static dispatch remains possible.
function CoDual(x::Type{P}, dx::NoFData) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x),NoFData}(P, dx)
end

function CoDual(x::Type{P}, dx::NoTangent) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(P, dx)
end

primal(x::CoDual) = x.x
tangent(x::CoDual) = x.dx
Base.copy(x::CoDual) = CoDual(copy(primal(x)), copy(tangent(x)))
_copy(x::P) where {P<:CoDual} = x

"""
    zero_codual(x)

Equivalent to `CoDual(x, zero_tangent(x))`.
"""
zero_codual(x) = CoDual(x, zero_tangent(x))

"""
    uninit_codual(x)

Equivalent to `CoDual(x, uninit_tangent(x))`.
"""
uninit_codual(x) = CoDual(x, uninit_tangent(x))

"""
    codual_type(P::Type)

The type of the `CoDual` which contains instances of `P` and associated tangents.
"""
function codual_type(::Type{P}) where {P}
    P == DataType && return CoDual
    P isa Union && return Union{codual_type(P.a),codual_type(P.b)}

    if P <: Tuple && !all(isconcretetype, P.parameters)
        field_types = (P.parameters..., )
        union_fields = findall(Base.Fix2(isa, Union), field_types)
        if length(union_fields) == 1 && all(p -> p isa Union || isconcretetype(p), field_types)
            P_split = split_tangent_type(field_types)
            return Union{codual_type(P_split.a), codual_type(P_split.b)}
        end
    end

    P <: UnionAll && return CoDual # P is abstract, so we don't know its tangent type.
    return isconcretetype(P) ? CoDual{P,tangent_type(P)} : CoDual
end

function codual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? CoDual{Type{P},NoTangent} : CoDual{_typeof(p),NoTangent}
end

struct NoPullback{R<:Tuple}
    r::R
end

_copy(x::P) where {P<:NoPullback} = P(_copy(x.r))

"""
    NoPullback(args::CoDual...)

Construct a `NoPullback` from the arguments passed to an `rrule!!`. For each argument,
extracts the primal value, and constructs a `LazyZeroRData`. These are stored in a
`NoPullback` which, in the reverse-pass of AD, instantiates these `LazyZeroRData`s and
returns them in order to perform the reverse-pass of AD.

The advantage of this approach is that if it is possible to construct the zero rdata element
for each of the arguments lazily, the `NoPullback` generated will be a singleton type. This
means that AD can avoid generating a stack to store this pullback, which can result in
significant performance improvements.
"""
function NoPullback(args::Vararg{CoDual,N}) where {N}
    return NoPullback(tuple_map(lazy_zero_rdata ∘ primal, args))
end

@inline (pb::NoPullback)(_) = tuple_map(instantiate, pb.r)

to_fwds(x::CoDual) = CoDual(primal(x), fdata(tangent(x)))

to_fwds(x::CoDual{Type{P}}) where {P} = CoDual{Type{P},NoFData}(primal(x), NoFData())

zero_fcodual(p) = to_fwds(zero_codual(p))

"""
    uninit_fcodual(x)

Like `zero_fcodual`, but doesn't guarantee that the value of the fdata is initialised.
See implementation for details, as this function is subject to change.
"""
@inline uninit_fcodual(x::P) where {P} = CoDual(x, uninit_fdata(x))

"""
    fcodual_type(P::Type)

The type of the `CoDual` which contains instances of `P` and its fdata.
"""
@inline function fcodual_type(::Type{P}) where {P}

    P == Union{} && return CoDual

    P == DataType && return CoDual

    # Small Unions of types result in small unions of codual types.
    P isa Union && return Union{fcodual_type(P.a),fcodual_type(P.b)}

    if P <: Tuple && !all(isconcretetype, (P.parameters..., ))
        field_types = (P.parameters..., )
        union_fields = _findall(Base.Fix2(isa, Union), 1, field_types)
        if length(union_fields) == 1 && all(p -> p isa Union || isconcretetype(p), field_types)
            P_split = split_tangent_type(field_types)
            return Union{fcodual_type(P_split.a), fcodual_type(P_split.b)}
        end
    end

    # If `P` is a UnionAll, then give up.
    P <: UnionAll && return CoDual

    # If `P` is a concrete type, we can do accurate inference. Otherwise give up.
    return isconcretetype(P) ? CoDual{P,fdata_type(tangent_type(P))} : CoDual
end

function fcodual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? CoDual{Type{P},NoFData} : CoDual{_typeof(p),NoFData}
end
