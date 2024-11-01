using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake, SpecialFunctions, Test

# Rules in this file are only lightly tester, because they are all just @from_rrule rules.
@testset "special_functions" begin
    @testset for (perf_flag, f, x...) in [
        (:stability, airyai, 0.1),
        (:stability, airyaix, 0.1),
        (:stability, airyaiprime, 0.1),
        (:stability, airybi, 0.1),
        (:stability, airybiprime, 0.1),
        (:stability_and_allocs, besselj0, 0.1),
        (:stability_and_allocs, besselj1, 0.1),
        (:stability_and_allocs, bessely0, 0.1),
        (:stability_and_allocs, bessely1, 0.1),
        (:stability_and_allocs, dawson, 0.1),
        (:stability_and_allocs, digamma, 0.1),
        (:stability_and_allocs, erf, 0.1),
        (:stability_and_allocs, erf, 0.1, 0.5),
        (:stability_and_allocs, erfc, 0.1),
        (:stability_and_allocs, logerfc, 0.1),
        (:stability_and_allocs, erfcinv, 0.1),
        (:stability_and_allocs, erfcx, 0.1),
        (:stability_and_allocs, logerfcx, 0.1),
        (:stability_and_allocs, erfi, 0.1),
        (:stability_and_allocs, erfinv, 0.1),
        (:stability_and_allocs, gamma, 0.1),
        (:stability_and_allocs, invdigamma, 0.1),
        (:stability_and_allocs, trigamma, 0.1),
        (:stability_and_allocs, polygamma, 3, 0.1),
        (:stability_and_allocs, beta, 0.3, 0.1),
        (:stability_and_allocs, logbeta, 0.3, 0.1),
        (:stability_and_allocs, logabsgamma, 0.3),
        (:stability_and_allocs, loggamma, 0.3),
        (:stability_and_allocs, expint, 0.3),
        (:stability_and_allocs, expintx, 0.3),
        (:stability_and_allocs, expinti, 0.3),
        (:stability_and_allocs, sinint, 0.3),
        (:stability_and_allocs, cosint, 0.3),
        (:stability_and_allocs, ellipk, 0.3),
        (:stability_and_allocs, ellipe, 0.3),
        (:stability_and_allocs, logfactorial, 3),
    ]
        test_rule(Xoshiro(123456), f, x...; perf_flag)
    end
    @testset for (perf_flag, f, x...) in [
        (:allocs, logerf, 0.3, 0.5), # first branch
        (:allocs, logerf, 1.1, 1.2), # second branch
        (:allocs, logerf, -1.2, -1.1), # third branch
        (:allocs, logerf, 0.3, 1.1), # fourth branch
        (:allocs, SpecialFunctions.loggammadiv, 1.0, 9.0),
        (:allocs, SpecialFunctions.gammax, 1.0),
        (:allocs, SpecialFunctions.rgammax, 3.0, 6.0),
        (:allocs, SpecialFunctions.rgamma1pm1, 0.1),
        (:allocs, SpecialFunctions.auxgam, 0.1),
        (:allocs, logabsbeta, 0.3, 0.1),
        (:allocs, SpecialFunctions.loggamma1p, 0.3),
        (:allocs, SpecialFunctions.loggamma1p, -0.3),
        (:none, SpecialFunctions.lambdaeta, 5.0),
    ]
        test_rule(Xoshiro(123456), f, x...; perf_flag, is_primitive=false)
    end
end
