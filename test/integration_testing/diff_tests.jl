@testset "diff_tests" begin
    @testset "$f, $(typeof(x))" for (interface_only, f, x...) in vcat(
        TestResources.DIFFTESTS_FUNCTIONS[1:66], # SKIPPING SPARSE_LDIV
        TestResources.DIFFTESTS_FUNCTIONS[68:89], # SKIPPING SPARSE_LDIV
        TestResources.DIFFTESTS_FUNCTIONS[91:end], # SKIPPING SPARSE_LDIV
    )
        @info "$(map(typeof, (f, x...)))"
        rng = Xoshiro(123456)
        TestUtils.test_interpreted_rrule!!(rng, f, deepcopy(x)...; interface_only, perf_flag=:none)
        # test_taped_rrule!!(rng, f, deepcopy(x)...; interface_only, perf_flag=:none)
    end
end
