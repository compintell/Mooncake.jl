module TapirNNlibExt

    using NNlib, Tapir
    using Base: IEEEFloat

    import Tapir: @from_rrule, DefaultCtx

    @from_rrule(
        DefaultCtx, Tuple{typeof(batched_mul), Array{<:IEEEFloat, 3}, Array{<:IEEEFloat, 3}}
    )
    @from_rrule(
        DefaultCtx,
        Tuple{typeof(Core.kwcall), NamedTuple, typeof(softmax), Array{<:IEEEFloat}},
    )
    @from_rrule(
        DefaultCtx,
        Tuple{typeof(Core.kwcall), NamedTuple, typeof(logsoftmax), Array{<:IEEEFloat}},
    )
    @from_rrule(
        DefaultCtx,
        Tuple{typeof(Core.kwcall), NamedTuple, typeof(logsumexp), Array{<:IEEEFloat}},
    )
    @from_rrule(
        DefaultCtx,
        Tuple{typeof(upsample_nearest), Array{<:IEEEFloat}, NTuple{N, Int} where {N}},
    )
end
