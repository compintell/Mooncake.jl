using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DynamicPPL, Mooncake, StableRNGs, Test
using DynamicPPL: istrans, VarInfo

@testset "DynamicPPLMooncakeExt" begin
    Mooncake.TestUtils.test_rule(StableRNG(123456), istrans, VarInfo(); unsafe_perturb=true)
end
