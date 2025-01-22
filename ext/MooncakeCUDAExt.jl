module MooncakeCUDAExt

using LinearAlgebra, Random, Mooncake

using Base: IEEEFloat
using CUDA: CuArray, cu

import Mooncake:
    MinimalCtx,
    rrule!!,
    @is_primitive,
    tangent_type,
    tangent,
    zero_tangent_internal,
    randn_tangent_internal,
    increment_internal!!,
    set_to_zero_internal!!,
    _add_to_primal_internal,
    _diff_internal,
    _dot_internal,
    _scale_internal,
    TestUtils,
    CoDual,
    NoPullback,
    to_cr_tangent,
    increment_and_get_rdata!,
    MaybeCache,
    IncCache

import Mooncake.TestUtils: populate_address_map_internal, AddressMap, __increment_should_allocate

# Tell Mooncake.jl how to handle CuArrays.

Mooncake.@tt_effects tangent_type(::Type{P}) where {P<:CuArray{<:IEEEFloat}} = P
function zero_tangent_internal(x::P, stackdict::Any) where {P<:CuArray{<:IEEEFloat}}
    haskey(stackdict, x) && return stackdict[x]::tangent_type(P)
    t = zero(x)
    stackdict[x] = t
    return t
end
function randn_tangent_internal(rng::AbstractRNG, x::P, stackdict::Any) where {P<:CuArray{<:IEEEFloat}}
    haskey(stackdict, x) && return stackdict[x]::tangent_type(P)
    t = cu(randn(rng, Float32, size(x)...))
    stackdict[x] = t
    return t
end
function TestUtils.has_equal_data_internal(
    x::P, y::P, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {P<:CuArray{<:IEEEFloat}}
    return isapprox(x, y)
end
function increment_internal!!(c::IncCache, x::P, y::P) where {P<:CuArray{<:IEEEFloat}}
    (x === y || haskey(c, x)) && return x
    c[x] = true
    x .+= y
    return x
end
__increment_should_allocate(::Type{<:CuArray{<:IEEEFloat}}) = true
set_to_zero_internal!!(::Mooncake.IncCache, x::CuArray{<:IEEEFloat}) = x .= 0
function _add_to_primal_internal(c::MaybeCache, x::P, y::P, unsafe::Bool) where {P<:CuArray{<:IEEEFloat}}
    key = (x, y, unsafe)
    haskey(c, key) && return c[key]::P
    x′ = x + y
    c[(x, y, unsafe)] = x′
    return x′
end
function _diff_internal(c::MaybeCache, x::P, y::P) where {P<:CuArray{<:IEEEFloat}}
    key = (x, y)
    haskey(c, key) && return c[key]::tangent_type(P)
    t = x - y
    c[key] = t
    return t
end
function _dot_internal(c::MaybeCache, x::P, y::P) where {P<:CuArray{<:IEEEFloat}}
    key = (x, y)
    haskey(c, key) && return c[key]::Float64
    return Float64(dot(x, y))
end
function _scale_internal(c::MaybeCache, x::Float64, y::P) where {T<:IEEEFloat,P<:CuArray{T}}
    haskey(c, y) && return c[y]::P
    t′ = T(x) * y
    c[y] = t′
    return t′
end
function populate_address_map_internal(m::AddressMap, p::CuArray, t::CuArray)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    return m
end
function Mooncake.__verify_fdata_value(::IdDict{Any,Nothing}, p::CuArray, f::CuArray)
    if size(p) != size(f)
        throw(InvalidFDataException("p has size $(size(p)) but f has size $(size(f))"))
    end
    return nothing
end
tangent_type(::Type{P}, ::Type{NoRData}) where {P<:CuArray} = P
tangent(p::CuArray, ::NoRData) = p

to_cr_tangent(x::CuArray{<:IEEEFloat}) = x
function increment_and_get_rdata!(f::T, ::NoRData, t::T) where {T<:CuArray{<:IEEEFloat}}
    f .+= t
    return NoRData()
end

# Basic rules for operating on CuArrays.

@is_primitive(MinimalCtx, Tuple{Type{<:CuArray},UndefInitializer,Vararg{Int,N}} where {N},)
function rrule!!(
    p::CoDual{Type{P}}, init::CoDual{UndefInitializer}, dims::CoDual{Int}...
) where {P<:CuArray{<:Base.IEEEFloat}}
    _dims = map(primal, dims)
    return CoDual(P(undef, _dims), P(undef, _dims)), NoPullback(p, init, dims...)
end

end
