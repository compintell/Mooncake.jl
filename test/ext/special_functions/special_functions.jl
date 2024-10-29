using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake, SpecialFunctions, Test

@testset "special_functions" begin
    @testset for (perf_flag, f, x...) in [
        (:stability, airyai, 0.1),
        (:stability, airyai, 0.0),
        (:stability, airyai, -0.5),
        (:stability, airyaix, 0.1),
        (:stability, airyaix, 0.05),
        (:stability, airyaix, 0.9),
        (:stability_and_allocs, erfc, 0.1),
        (:stability_and_allocs, erfc, 0.0),
        (:stability_and_allocs, erfc, -0.5),
        (:stability_and_allocs, erfcx, 0.1),
        (:stability_and_allocs, erfcx, 0.0),
        (:stability_and_allocs, erfcx, -0.5),
    ]
        test_rule(Xoshiro(123456), f, x...; perf_flag)
    end
end
