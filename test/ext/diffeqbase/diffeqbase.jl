using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, JET
using Mooncake, StableRNGs, Test, DiffEqBase
using DiffEqBase: SciMLBase
using Mooncake.TestUtils: test_rule

@testset "diffeqbase" begin
    @testset "$f, $(typeof(fargs))" for (
        interface_only, perf_flag, is_primitive, f, fargs
    ) in [(
        false,
        :stability_and_allocs,
        true,
        DiffEqBase.set_mooncakeoriginator_if_mooncake,
        SciMLBase.ChainRulesOriginator(),
    )]
        test_rule(StableRNG(123), f, fargs; interface_only, perf_flag, is_primitive)
    end
end