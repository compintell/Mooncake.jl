using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DynamicPPL, Mooncake, Test

@testset "DynamicPPLMooncakeExt" begin
    test_rule(sr(123456), DynamicPPL.istrans, DynamicPPL.VarInfo(); unsafe_perturb=true)
end
