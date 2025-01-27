# include("../front_matter.jl")

#=
Failing cases:
- 10: need help with a PossiblyUninitTangent in the input duals
=#
working_cases = vcat(1:10)

@testset verbose = true "s2s_forward_mode_ad" begin
    test_cases = collect(enumerate(TestResources.generate_test_functions()))[working_cases]
    @testset "$(_typeof((f, x...)))" for (n, (int_only, pf, _, f, x...)) in test_cases
        sig = _typeof((f, x...))
        @info "$n: $sig"
        TestUtils.test_rule(
            Xoshiro(123456),
            f,
            x...;
            perf_flag=pf,
            interface_only=int_only,
            is_primitive=false,
            forward=true,
        )
    end
end;
