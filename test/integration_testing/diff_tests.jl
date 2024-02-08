@testset "diff_tests" begin
    interp = Taped.TInterp()
    @testset "$f, $(_typeof(x))" for (interface_only, f, x...) in vcat(
        TestResources.DIFFTESTS_FUNCTIONS[1:31], # SKIPPING SPARSE_LDIV mat2num_4 and softmax due to `_apply_iterate` handling
        TestResources.DIFFTESTS_FUNCTIONS[34:66], # SKIPPING SPARSE_LDIV
        TestResources.DIFFTESTS_FUNCTIONS[68:89], # SKIPPING SPARSE_LDIV
        TestResources.DIFFTESTS_FUNCTIONS[91:end], # SKIPPING SPARSE_LDIV
    )
        @info "$(_typeof((f, x...)))"
        TestUtils.test_interpreted_rrule!!(
            sr(123456), f, x...;
            interp, perf_flag=:none, interface_only=false, is_primitive=false,
        )
    end
end
