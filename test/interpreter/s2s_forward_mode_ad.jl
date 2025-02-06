include("../front_matter.jl")

#=
Failing cases:
- 4: fails an allocation test on the first run only (rule compilation?)
- 17: rule for `new` with uninitialized tangents for some fields
- 28: stackoverflow (probably in method recognition and rule insertion?)
- 32: rule for `new` with namedtuple is incorrect
=#
working_cases = vcat(1:16, 18:27, 29:31)

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
