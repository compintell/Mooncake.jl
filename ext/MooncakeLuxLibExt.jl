module MooncakeLuxLibExt

    using LuxLib, Random, Mooncake
    using Base: IEEEFloat

    import LuxLib.Impl: matmul, matmuladd, fused_dense
    import Mooncake: @from_rrule, DefaultCtx

    @from_rrule(DefaultCtx, Tuple{typeof(matmul), Array{P}, Array{P}} where {P<:IEEEFloat})
    @from_rrule(
        DefaultCtx,
        Tuple{typeof(matmuladd), Array{P}, Array{P}, Vector{P}} where {P<:IEEEFloat},
    )

    # The implementations of rrules for fused operations are not straightforward to
    # incorporate into Mooncake.jl, because they call back into AD.
    # We take a simple appoach to their implementation: differentiate an un-fused version
    # of their implementation. This will likely hit performance, but it makes implementing
    # rules much more straightforward, in that we only have to be able to implement their
    # constituent parts, rather than the entire thing.
end
