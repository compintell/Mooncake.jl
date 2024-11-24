module MooncakeNNlibExt

using NNlib, Random, Mooncake
using Base: IEEEFloat
using NNlib: dropout

using NNlib: conv, depthwiseconv
import Mooncake: @from_rrule, DefaultCtx, MinimalCtx

@from_rrule(
    MinimalCtx, Tuple{typeof(batched_mul),Array{P,3},Array{P,3}} where {P<:IEEEFloat},
)
@from_rrule(
    MinimalCtx, Tuple{typeof(dropout),AbstractRNG,Array{P},P} where {P<:IEEEFloat}, true,
)
@from_rrule(MinimalCtx, Tuple{typeof(softmax),Array{<:IEEEFloat}}, true)
@from_rrule(MinimalCtx, Tuple{typeof(logsoftmax),Array{<:IEEEFloat}}, true)
@from_rrule(MinimalCtx, Tuple{typeof(logsumexp),Array{<:IEEEFloat}}, true)
@from_rrule(
    MinimalCtx, Tuple{typeof(upsample_nearest),Array{<:IEEEFloat},NTuple{N,Int} where {N}},
)
@from_rrule(
    MinimalCtx,
    Tuple{typeof(NNlib.fold),Array{<:IEEEFloat},NTuple{N,Int} where {N},DenseConvDims},
)
@from_rrule(MinimalCtx, Tuple{typeof(NNlib.unfold),Array{<:IEEEFloat},DenseConvDims})
@from_rrule(
    MinimalCtx, Tuple{typeof(NNlib.scatter),Any,Array,Array{<:Union{Integer,Tuple}}}, true,
)
for conv in [:conv, :depthwiseconv]
    local ∇conv_data, ∇conv_filter = Symbol.(:∇, conv, [:_data, :_filter])

    @eval @from_rrule(
        MinimalCtx,
        Tuple{typeof($conv),Array{P},Array{P},ConvDims} where {P<:IEEEFloat},
        true,
    )
    @eval @from_rrule(
        MinimalCtx,
        Tuple{typeof($∇conv_data),Array{P},Array{P},ConvDims} where {P<:IEEEFloat},
        true,
    )
end
@from_rrule(
    MinimalCtx,
    Tuple{typeof(∇conv_filter),Array{P},Array{P},ConvDims} where {P<:IEEEFloat},
    true,
)
for pool in [:maxpool, :meanpool]
    @eval @from_rrule(MinimalCtx, Tuple{typeof($pool),Array{<:IEEEFloat},PoolDims}, true)
end
@from_rrule(MinimalCtx, Tuple{typeof(pad_constant),Array,Any,Any}, true)

end
