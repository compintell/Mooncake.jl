using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake, StableRNGs, Test
using Flux: Losses
using Mooncake.TestUtils: test_rule

@testset "flux" begin
    @testset "$f, $(_typeof(fargs))" for (
        interface_only, perf_flag, is_primitive, f, fargs...
    ) in vcat(
        # Testing specific floating point precisions as FDM is unstable for certain tests.
        # Larger differences can compare well with AD for for higher precisions as FDM is relatively stable.
        # for smaller differences we start getting better FDM gradients for smaller floating points.

        map([Float64]) do P
            return (
                false, :none, nothing, Losses.mse, P.([1e1, 1e2, 1e3]), P.([1e3, 1e4, 0])
            )
        end,
        map([Float32, Float64]) do P
            return (
                false,
                :none,
                nothing,
                Losses.mse,
                Float64.([13, 13, 13]),
                Float64.([13, 13, 13]),
            )
        end,
        map([Float16, Float32, Float64]) do P
            return (
                false,
                :none,
                nothing,
                Losses.mse,
                P.([1e-3, 1e-4, 0]),
                P.([1e-1, 1e-2, 1e-3]),
            )
        end,
    )
        test_rule(StableRNG(123), f, fargs...; interface_only, perf_flag, is_primitive)
    end
end
