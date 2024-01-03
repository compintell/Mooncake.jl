@testset "diff_tests" begin
    interp = Taped.TInterp()
    @testset "$f, $(typeof(x))" for (interface_only, f, x...) in vcat(
        TestResources.DIFFTESTS_FUNCTIONS[1:66], # SKIPPING SPARSE_LDIV
        TestResources.DIFFTESTS_FUNCTIONS[68:89], # SKIPPING SPARSE_LDIV
        TestResources.DIFFTESTS_FUNCTIONS[91:end], # SKIPPING SPARSE_LDIV
    )
        @info "$(map(typeof, (f, x...)))"
        rng = Xoshiro(123456)
        sig = Tuple{Core.Typeof(f), map(Core.Typeof, x)...}
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp)
        if interface_only
            in_f(f, deepcopy(x)...)
        else
            x_cpy_1 = deepcopy(x)
            x_cpy_2 = deepcopy(x)
            @test has_equal_data(in_f(f, x_cpy_1...), f(x_cpy_2...))
            @test has_equal_data(x_cpy_1, x_cpy_2)
        end
        # TestUtils.test_interpreted_rrule!!(rng, f, deepcopy(x)...; interface_only, perf_flag=:none)
    end
end
