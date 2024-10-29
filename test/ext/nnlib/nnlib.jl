using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake, NNlib, Test
using NNlib: dropout

@testset "nnlib" begin
    x = randn(5, 4, 3, 2)
    w = randn(2, 2, 3, 3)
    dense_cdims = DenseConvDims(x, w)
    sep_cdims = DepthwiseConvDims(x, w)
    y = conv(x, w, dense_cdims)
    y_sep = depthwiseconv(x, w, sep_cdims)

    pool_dims = PoolDims(size(x), 2)

    grid = Array{Float64}(undef, 2, 2, 2, 1)
    grid[:, 1, 1, 1] .= (-1, -1)
    grid[:, 2, 1, 1] .= (1, -1)
    grid[:, 1, 2, 1] .= (-1, 1)
    grid[:, 2, 2, 1] .= (1, 1)

    @testset "$(typeof(fargs))" for (interface_only, perf_flag, is_primitive, fargs...) in Any[

        # batched_mul
        (false, :none, true, batched_mul, randn(3, 2, 3), randn(2, 5, 3)),

        # dropout
        (true, :none, false, (x, p) -> dropout(sr(1), x, p; dims=1), randn(2, 2), 0.5),
        (true, :none, false, (x, p) -> dropout(sr(1), x, p; dims=2), randn(2, 2), 0.1),
        (true, :none, false, (x, p) -> dropout(sr(1), x, p; dims=(1, 2)), randn(2, 2), 0.4),

        # softmax
        (false, :stability, true, softmax, randn(2)),
        (false, :stability, true, softmax, randn(2, 2)),
        (false, :stability, true, Core.kwcall, (dims=1, ), softmax, randn(2,)),
        (false, :stability, true, Core.kwcall, (dims=1, ), softmax, randn(3, 3)),
        (false, :stability, true, Core.kwcall, (dims=2, ), softmax, randn(3, 3)),
        (false, :stability, true, Core.kwcall, (dims=(1, 2), ), softmax, randn(3, 3)),
        (false, :stability, true, Core.kwcall, (dims=(1, 2), ), softmax, randn(3, 3, 2)),
        (false, :none, false, x -> softmax(5x), randn(3, 2)),
        (false, :none, false, x -> softmax(x; dims=1), randn(3, 2)),
        (false, :none, false, x -> softmax(x; dims=2), randn(3, 2)),
        (false, :none, false, x -> softmax(x; dims=(1, 2)), randn(3, 2)),

        # logsoftmax
        (false, :stability, true, logsoftmax, randn(2)),
        (false, :stability, true, logsoftmax, randn(2, 3)),
        (false, :stability, true, logsoftmax, randn(2, 3, 2)),
        (false, :stability, true, Core.kwcall, (dims=1, ), logsoftmax, randn(2,)),
        (false, :stability, true, Core.kwcall, (dims=1, ), logsoftmax, randn(3, 3)),
        (false, :stability, true, Core.kwcall, (dims=2, ), logsoftmax, randn(3, 3)),
        (false, :stability, true, Core.kwcall, (dims=(1, 2), ), logsoftmax, randn(3, 3)),
        (false, :stability, true, Core.kwcall, (dims=(1, 2), ), logsoftmax, randn(3, 3, 2)),

        # logsumexp
        (false, :stability, true, logsumexp, randn(2,)),
        (false, :stability, true, logsumexp, randn(3, 3)),
        (false, :stability, true, logsumexp, randn(3, 3, 2)),
        (false, :stability, true, Core.kwcall, (dims=1, ), logsumexp, randn(2,)),
        (false, :stability, true, Core.kwcall, (dims=1, ), logsumexp, randn(3, 3)),
        (false, :stability, true, Core.kwcall, (dims=2, ), logsumexp, randn(3, 3)),
        (false, :stability, true, Core.kwcall, (dims=(1, 2), ), logsumexp, randn(3, 3)),
        (false, :stability, true, Core.kwcall, (dims=(1, 2), ), logsumexp, randn(3, 3, 2)),

        # upsample_nearest
        (false, :stability, true, upsample_nearest, randn(3), (2,)),
        (false, :stability, true, upsample_nearest, randn(3, 2), (2, 2)),
        (false, :stability, true, upsample_nearest, randn(3, 2, 3), (2, 2, 5)),

        # fold
        (false, :none, true, NNlib.fold, randn(12, 12, 2), size(x), dense_cdims),

        # unfold
        (false, :none, true, NNlib.unfold, x, dense_cdims),

        # scatter
        (false, :stability, true, NNlib.scatter, +, randn(2), [1, 3]),
        (false, :stability, true, Core.kwcall, (;), NNlib.scatter, +, randn(2), [1, 3]),

        # conv
        (false, :none, true, Core.kwcall, (;), conv, x, w, dense_cdims),
        (false, :none, true, conv, x, w, dense_cdims),
        (false, :none, true, Core.kwcall, (;), depthwiseconv, x, w, sep_cdims),
        (false, :none, true, depthwiseconv, x, w, sep_cdims),

        # ∇conv_data
        (false, :none, true, Core.kwcall, (;), ∇conv_data, y, w, dense_cdims),
        (false, :none, true, ∇conv_data, y, w, dense_cdims),
        (false, :none, true, Core.kwcall, (;), ∇depthwiseconv_data, y_sep, w, sep_cdims),
        (false, :none, true, ∇depthwiseconv_data, y_sep, w, sep_cdims),

        # ∇conv_filter
        (false, :none, true, Core.kwcall, (;), ∇conv_filter, x, y, dense_cdims),
        (false, :none, true, ∇conv_filter, x, y, dense_cdims),

        # pooling
        (false, :none, true, maxpool, x, pool_dims),
        (false, :none, true, Core.kwcall, (;), maxpool, x, pool_dims),
        (false, :none, true, meanpool, x, pool_dims),
        (false, :none, true, Core.kwcall, (;), meanpool, x, pool_dims),

        # padding
        (false, :none, false, x -> pad_constant(x, 1, 2.0), x),
        (false, :none, false, x -> pad_constant(x, 1, 2.0; dims=:), x),
    ]
        @info "$(typeof(fargs))"
        test_rule(sr(1), fargs...; perf_flag, is_primitive, interface_only)
    end
end
