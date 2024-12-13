module MooncakeCUDAExt

using LinearAlgebra, Random, Mooncake

using Base: IEEEFloat
using CUDA: CuArray, cu

import Mooncake:
    MinimalCtx,
    rrule!!,
    @is_primitive,
    tangent_type,
    zero_tangent,
    randn_tangent,
    increment!!,
    _set_to_zero!!,
    __add_to_primal,
    __diff,
    __dot,
    __scale,
    TestUtils,
    CoDual,
    NoPullback

import Mooncake.TestUtils: populate_address_map!, AddressMap, __increment_should_allocate

# Tell Mooncake.jl how to handle CuArrays.

tangent_type(::Type{P}) where {P<:CuArray{<:IEEEFloat}} = P
zero_tangent(x::CuArray{<:IEEEFloat}) = zero(x)
function randn_tangent(rng::AbstractRNG, x::CuArray{Float32})
    return cu(randn(rng, Float32, size(x)...))
end
TestUtils.has_equal_data(x::P, y::P) where {P<:CuArray{<:IEEEFloat}} = x == y
increment!!(x::P, y::P) where {P<:CuArray{<:IEEEFloat}} = x .+= y
__increment_should_allocate(::Type{<:CuArray{<:IEEEFloat}}) = true
_set_to_zero!!(::Mooncake.IncCache, x::CuArray{<:IEEEFloat}) = x .= 0
function __add_to_primal(c::MaybeCache, x::P, y::P, ::Bool) where {P<:CuArray{<:IEEEFloat}}
    key = (x, y, unsafe)
    haskey(c, key) && return c[key]::P
    x′ = x + y
    c[(x, y, unsafe)] = x′
    return x′
end
function __diff(c::Cache, x::P, y::P) where {P<:CuArray{<:IEEEFloat}}
    key = (x, y)
    haskey(c, key) && return c[key]::tangent_type(P)
    t = x - y
    c[key] = t
    return t
end
function __dot(c::MaybeCache, x::P, y::P) where {P<:CuArray{<:IEEEFloat}}
    key = (x, y)
    haskey(c, key) && return c[key]::Float64
    return Float64(dot(x, y))
end
function __scale(c::MaybeCache, x::Float64, y::P) where {T<:IEEEFloat,P<:CuArray{T}}
    haskey(c, y) && return c[y]::P
    t′ = T(x) * y
    c[y] = t′
    return t′
end
function populate_address_map!(m::AddressMap, p::CuArray, t::CuArray)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    return m
end
function Mooncake._verify_fdata_value(p::CuArray, f::CuArray)
    if size(p) != size(f)
        throw(InvalidFDataException("p has size $(size(p)) but f has size $(size(f))"))
    end
    return nothing
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
