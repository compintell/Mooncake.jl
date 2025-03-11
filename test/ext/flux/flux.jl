using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake, StableRNGs, Test, Flux
using Mooncake.TestUtils: test_rule

@testset "flux" begin
    @testset "$f, $(typeof(fargs))" for (
        interface_only, perf_flag, is_primitive, f, fargs...
    ) in vcat(
        # Testing specific floating point precisions as FDM is unstable for certain tests.
        # Larger differences can compare well with AD only for higher precisions as FDM is relatively stable.
        # for smaller differences we start getting better FDM gradients for smaller floating points as well.

        map([Float32, Float64]) do P
            return (
                false,
                :none,
                false,
                Flux.Losses.mse,
                randn(StableRNG(1), P, 3),
                randn(StableRNG(2), P, 3);
            )
        end,
    )
        test_rule(StableRNG(123), f, fargs...; interface_only, perf_flag, is_primitive)
    end
end
