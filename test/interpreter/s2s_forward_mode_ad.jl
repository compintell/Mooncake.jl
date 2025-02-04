include("../front_matter.jl")

#=
Failing cases:
- 4: fails an allocation test on the first run only (rule compilation?)
- 15,16,17,18: I don't understand why `DataType` doesn't dispatch on `Type{P}`
- 23: segfault on `GlobalRef`
=#
working_cases = vcat(1:14, 19:22)

@testset verbose = true "s2s_forward_mode_ad" begin
    test_cases = collect(enumerate(TestResources.generate_test_functions()))[working_cases]
    @testset "$n - $(_typeof((f, x...)))" for (n, (int_only, pf, _, f, x...)) in test_cases
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
