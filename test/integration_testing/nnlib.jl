@testset "nnlib" begin
    @testset "$(typeof(fargs))" for (perf_flag, fargs...) in Any[

        # batched_mul
        (:none, batched_mul, randn(3, 2, 3), randn(2, 5, 3)),

        # softmax
        (:stability, Core.kwcall, (dims=1, ), softmax, randn(2,)),
        (:stability, Core.kwcall, (dims=1, ), softmax, randn(3, 3)),
        (:stability, Core.kwcall, (dims=2, ), softmax, randn(3, 3)),
        (:stability, Core.kwcall, (dims=(1, 2), ), softmax, randn(3, 3)),
        (:stability, Core.kwcall, (dims=(1, 2), ), softmax, randn(3, 3, 2)),
        (:none, x -> softmax(5x), randn(3, 2)),
        (:none, x -> softmax(x; dims=1), randn(3, 2)),
        (:none, x -> softmax(x; dims=2), randn(3, 2)),
        (:none, x -> softmax(x; dims=(1, 2)), randn(3, 2)),

        # logsoftmax
        (:stability, Core.kwcall, (dims=1, ), logsoftmax, randn(2,)),
        (:stability, Core.kwcall, (dims=1, ), logsoftmax, randn(3, 3)),
        (:stability, Core.kwcall, (dims=2, ), logsoftmax, randn(3, 3)),
        (:stability, Core.kwcall, (dims=(1, 2), ), logsoftmax, randn(3, 3)),
        (:stability, Core.kwcall, (dims=(1, 2), ), logsoftmax, randn(3, 3, 2)),

        # logsumexp
        (:stability, Core.kwcall, (dims=1, ), logsumexp, randn(2,)),
        (:stability, Core.kwcall, (dims=1, ), logsumexp, randn(3, 3)),
        (:stability, Core.kwcall, (dims=2, ), logsumexp, randn(3, 3)),
        (:stability, Core.kwcall, (dims=(1, 2), ), logsumexp, randn(3, 3)),
        (:stability, Core.kwcall, (dims=(1, 2), ), logsumexp, randn(3, 3, 2)),

        # upsample_nearest
        (:stability, upsample_nearest, randn(3), (2,)),
        (:stability, upsample_nearest, randn(3, 2), (2, 2)),
        (:stability, upsample_nearest, randn(3, 2, 3), (2, 2, 5)),
    ]
        test_rule(sr(1), fargs...; is_primitive=false, perf_flag)
    end
end
