using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path = joinpath(@__DIR__, "..", "..", ".."))

using Mooncake, Random, StableRNGs, Test
using Mooncake.TestUtils: test_rule

@testset "diff_tests" begin
    @testset "$f, $(typeof(x))" for (n, (interface_only, f, x...)) in enumerate(vcat(
        Mooncake.TestResources.DIFFTESTS_FUNCTIONS[1:6], # skipping DiffTests.num2arr_1. See https://github.com/JuliaLang/julia/issues/56193
        Mooncake.TestResources.DIFFTESTS_FUNCTIONS[8:66], # skipping sparse_ldiv
        Mooncake.TestResources.DIFFTESTS_FUNCTIONS[68:89], # skipping sparse_ldiv
        Mooncake.TestResources.DIFFTESTS_FUNCTIONS[91:end], # skipping sparse_ldiv
    ))
        @info "$n: $(typeof((f, x...)))"
        test_rule(StableRNG(123456), f, x...; is_primitive=false)
    end
end
