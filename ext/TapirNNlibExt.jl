module TapirNNlibExt

    using NNlib, Random, Tapir
    using Base: IEEEFloat
    using NNlib: dropout

    import Tapir: @from_rrule, DefaultCtx

    @from_rrule(
        DefaultCtx,
        Tuple{typeof(batched_mul), Array{P, 3}, Array{P, 3}} where {P<:IEEEFloat},
    )
    @from_rrule(
        DefaultCtx,
        Tuple{typeof(dropout), AbstractRNG, Array{P}, P} where {P<:IEEEFloat},
        true,
    )
    @from_rrule(DefaultCtx, Tuple{typeof(softmax), Array{<:IEEEFloat}}, true)
    @from_rrule(DefaultCtx, Tuple{typeof(logsoftmax), Array{<:IEEEFloat}}, true)
    @from_rrule(DefaultCtx, Tuple{typeof(logsumexp), Array{<:IEEEFloat}}, true)
    @from_rrule(
        DefaultCtx,
        Tuple{typeof(upsample_nearest), Array{<:IEEEFloat}, NTuple{N, Int} where {N}},
    )
    @from_rrule(
        DefaultCtx,
        Tuple{
            typeof(NNlib.fold), Array{<:IEEEFloat}, NTuple{N, Int} where {N}, DenseConvDims,
        },
    )
    @from_rrule(
        DefaultCtx, Tuple{typeof(NNlib.unfold), Array{<:IEEEFloat}, DenseConvDims}
    )
    @from_rrule(
        DefaultCtx,
        Tuple{typeof(NNlib.scatter), Any, Array, Array{<:Union{Integer, Tuple}}},
        true,
    )
    for backend in (Symbol(), :_direct, :_im2col), name in (:conv, :depthwiseconv)
        @eval @from_rrule(
            DefaultCtx,
            Tuple{
                typeof(NNlib.$(Symbol("$name$(backend)"))), Array{P}, Array{P}, ConvDims,
            } where {P<:IEEEFloat},
            true,
        )
    end
    for pool in [:maxpool, :meanpool]
        @eval @from_rrule(
            DefaultCtx, Tuple{typeof($pool), Array{<:IEEEFloat}, PoolDims}, true
        )
    end
    @from_rrule(DefaultCtx, Tuple{typeof(pad_constant), Array, Any, Any}, true)
end
