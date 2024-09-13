module TapirNNlibExt

    using NNlib, Tapir
    using Base: IEEEFloat

    import Tapir: @from_rrule, DefaultCtx

    @from_rrule(
        DefaultCtx,
        Tuple{typeof(upsample_nearest), Array{<:IEEEFloat}, NTuple{N, Int} where {N}},
    )
end
