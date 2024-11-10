@testset "diff_tests" begin
    @testset "$f, $(_typeof(x))" for (n, (interface_only, f, x...)) in enumerate(vcat(
        TestResources.DIFFTESTS_FUNCTIONS[1:6], # skipping DiffTests.num2arr_1. See https://github.com/JuliaLang/julia/issues/56193
        TestResources.DIFFTESTS_FUNCTIONS[8:66], # skipping sparse_ldiv
        TestResources.DIFFTESTS_FUNCTIONS[68:89], # skipping sparse_ldiv
        TestResources.DIFFTESTS_FUNCTIONS[91:end], # skipping sparse_ldiv
    ))
        @info "$n: $(_typeof((f, x...)))"
        test_rule(sr(123456), f, x...; is_primitive=false)
    end
end
