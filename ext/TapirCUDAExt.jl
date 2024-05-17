module TapirCUDAExt

    using CUDA, Tapir
    using Base: IEEEFloat

    import Tapir: MinimalCtx, rrule!!, @is_primitive, tangent_type, CoDual

    Tapir.tangent_type(::Type{P}) where {P<:CuArray{<:IEEEFloat}} = P
    Tapir.zero_tangent(x::CuArray{<:IEEEFloat}) = zero(x)
    function Tapir.randn_tangent(rng::AbstractRNG, x::CuArray{Float32})
        return cu(randn(rng, size(x)...))
    end
    Tapir.TestUtils.has_equal_data(x::P, y::P) where {P<:CuArray{<:IEEEFloat}} = x == y
    Tapir.increment!!(x::P, y::P) where {P<:CuArray{<:IEEEFloat}} = x .+= y
    Tapir.set_to_zero!!(x::CuArray{<:IEEEFloat}) = x .= 0
    Tapir._add_to_primal(x::P, y::P) where {P<:CuArray{<:IEEEFloat}} = x + y
    Tapir._diff(x::P, y::P) where {P<:CuArray{<:IEEEFloat}} = x - y
    Tapir._dot(x::P, y::P) where {P<:CuArray{<:IEEEFloat}} = Float64(dot(x, y))
    Tapir._scale(x::Float64, y::P) where {T<:IEEEFloat, P<:CuArray{T}} = T(x) * y
    function Tapir.TestUtils.populate_address_map!(m::AddressMap, p::CuArray, t::CuArray)
        k = pointer_from_objref(p)
        v = pointer_from_objref(t)
        haskey(m, k) && (@assert m[k] == v)
        m[k] = v
        return m
    end

    println("loading ext")
    @is_primitive MinimalCtx Tuple{Type{<:CuArray}, UndefInitializer, Vararg{Int, N}} where {N}
    function rrule!!(
        ::CoDual{Type{P}},
        ::CoDual{UndefInitializer},
        dims::CoDual{Int}...
    ) where {P<:CuArray{<:Base.IEEEFloat}}
        y = CoDual(P(undef, dims), P(undef, dims))
        return y, NoPullback()
    end
end
