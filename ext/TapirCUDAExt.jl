module TapirCUDAExt

    using CUDA, Tapir

    import Tapir: MinimalCtx, rrule!!

    @is_primitive MinimalCtx Tuple{Type{<:CuArray}, UndefInitializer, Vararg{Int, N}}
    function rrule!!(
        ::CoDual{Type{C}},
        ::CoDual{UndefInitializer},
        dims::CoDual{Int}...
    ) where {P<:CuArray{<:IEEEFloat}}
        y = CoDual(P(undef, dims), P(undef, dims))
        return y, NoPullback()
    end
end
