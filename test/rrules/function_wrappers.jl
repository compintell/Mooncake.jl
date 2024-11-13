@testset "function_wrappers" begin
    rng = Xoshiro(123)
    p = FunctionWrapper{Float64, Tuple{Float64}}(sin)
    # TestUtils.test_tangent_consistency(rng, p)
    # TestUtils.test_fwds_rvs_data(rng, p)
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:function_wrappers))

end

# TODO:
# 1. Add proper test cases in the function_wrappers file (for both rrules).
# 2. Add performance tests for situations in which you already have the FunctionWrapper.
# 3. Check that repeated calls of a FunctionWrapper are fast once it is constructed.
