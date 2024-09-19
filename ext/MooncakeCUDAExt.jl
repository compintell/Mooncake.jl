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
    set_to_zero!!,
    _add_to_primal,
    _diff,
    _dot,
    _scale,
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
set_to_zero!!(x::CuArray{<:IEEEFloat}) = x .= 0
_add_to_primal(x::P, y::P) where {P<:CuArray{<:IEEEFloat}} = x + y
_diff(x::P, y::P) where {P<:CuArray{<:IEEEFloat}} = x - y
_dot(x::P, y::P) where {P<:CuArray{<:IEEEFloat}} = Float64(dot(x, y))
_scale(x::Float64, y::P) where {T<:IEEEFloat, P<:CuArray{T}} = T(x) * y
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

@is_primitive(
    MinimalCtx,
    Tuple{Type{<:CuArray}, UndefInitializer, Vararg{Int, N}} where {N},
)
function rrule!!(
    p::CoDual{Type{P}},
    init::CoDual{UndefInitializer},
    dims::CoDual{Int}...
) where {P<:CuArray{<:Base.IEEEFloat}}
    @show "a rule?"
    _dims = map(primal, dims)
    y = CoDual(P(undef, _dims), P(undef, _dims))
    return y, NoPullback(p, init, dims...)
end

end
