module TapirSpecialFunctionsExt

    using Adapt, Tapir

    import Tapir: MinimalCtx, rrule!!

    function rrule!!(
        ::CoDual{Type{C}},
        ::CoDual{UndefInitializer},
        dims::CoDual{Int}...
    ) where {P<:CUDA.CuArray{<:IEEEFloat}}
        y = CoDual(P(undef, dims), P(undef, dims))
        return y, NoPullback()
    end
end
