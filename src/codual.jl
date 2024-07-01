struct CoDual{Tx, Tdx}
    x::Tx
    dx::Tdx
end

# Always sharpen the first thing if it's a type so static dispatch remains possible.
function CoDual(x::Type{P}, dx::NoFData) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x), NoFData}(P, dx)
end

function CoDual(x::Type{P}, dx::NoTangent) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x), NoTangent}(P, dx)
end

primal(x::CoDual) = x.x
tangent(x::CoDual) = x.dx
Base.copy(x::CoDual) = CoDual(copy(primal(x)), copy(tangent(x)))

"""
    zero_codual(x)

Equivalent to `CoDual(x, zero_tangent(x))`.
"""
zero_codual(x) = CoDual(x, zero_tangent(x))

"""
    codual_type(P::Type)

The type of the `CoDual` which contains instances of `P` and associated tangents.
"""
function codual_type(::Type{P}) where {P}
    P == DataType && return CoDual
    P isa Union && return Union{codual_type(P.a), codual_type(P.b)}
    P <: UnionAll && return CoDual
    return isconcretetype(P) ? CoDual{P, tangent_type(P)} : CoDual
end

codual_type(::Type{Type{P}}) where {P} = CoDual{Type{P}, NoTangent}

struct NoPullback{R<:Tuple}
    r::R
end

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
function NoPullback(args::Vararg{CoDual, N}) where {N}
    return NoPullback(tuple_map(lazy_zero_rdata ∘ primal, args))
end

@inline (pb::NoPullback)(_) = tuple_map(instantiate, pb.r)

to_fwds(x::CoDual) = CoDual(primal(x), fdata(tangent(x)))

to_fwds(x::CoDual{Type{P}}) where {P} = CoDual{Type{P}, NoFData}(primal(x), NoFData())

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
function fcodual_type(::Type{P}) where {P}
    P == DataType && return CoDual
    P isa Union && return Union{fcodual_type(P.a), fcodual_type(P.b)}
    P <: UnionAll && return CoDual
    return isconcretetype(P) ? CoDual{P, fdata_type(tangent_type(P))} : CoDual
end

fcodual_type(::Type{Type{P}}) where {P} = CoDual{Type{P}, NoFData}

zero_rdata(x::CoDual) = zero_rdata(primal(x))
