module TapirCUDAExt

    using CUDA, Tapir

    import Tapir: MinimalCtx, rrule!!, @is_primitive, tangent_type, CoDual

    tangent_type(::Type{P}) where {P<:CuArray{<:Base.IEEEFloat}} = P

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
