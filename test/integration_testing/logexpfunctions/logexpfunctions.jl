using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, LogExpFunctions, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule

sr(n::Int) = StableRNG(n)

@testset "logexpfunctions" begin
    @testset for (perf_flag, f, x...) in [
        (:allocs, xlogx, 1.1),
        (:allocs, xlogy, 0.3, 1.2),
        (:allocs, xlog1py, 0.3, -0.5),
        (:allocs, xexpx, -0.5),
        (:allocs, xexpy, 1.0, -0.7),
        (:allocs, logistic, 0.5),
        (:allocs, logit, 0.3),
        (:allocs, logcosh, 1.5),
        (:allocs, logabssinh, 0.3),
        (:allocs, log1psq, 0.3),
        (:allocs, log1pexp, 0.1),
        (:allocs, log1mexp, -0.5),
        (:allocs, log2mexp, 0.1),
        (:allocs, logexpm1, 0.1),
        (:allocs, log1pmx, -0.95),
        (:allocs, logmxp1, 0.02),
        (:allocs, logaddexp, -0.5, 0.4),
        (:allocs, logsubexp, -0.5, -5.0),
        (:allocs, logsumexp, randn(sr(1), 5)),
        (:allocs, logsumexp, randn(sr(2), 5, 4)),
        (:allocs, logsumexp, randn(sr(3), 5, 4, 3)),
        (:none, x -> logsumexp(x; dims=1), randn(sr(4), 5, 4)),
        (:none, x -> logsumexp(x; dims=2), randn(sr(5), 5, 4)),
        (:none, logsumexp!, rand(sr(6), 5), randn(sr(7), 5, 4)),
        (:none, softmax, randn(sr(7), 10)),
        (:allocs, cloglog, 0.5),
        (:allocs, cexpexp, -0.3),
        (:allocs, loglogistic, 0.5),
        (:allocs, logitexp, -0.3),
        (:allocs, log1mlogistic, -0.9),
        (:allocs, logit1mexp, -0.6),
    ]
        test_rule(sr(123456), f, x...; perf_flag, is_primitive=false)
    end
end
