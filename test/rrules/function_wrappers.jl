@testset "function_wrappers" begin
    rng = Xoshiro(123)
    _data = Ref{Float64}(5.0)
    @testset "$p" for p in Any[
        FunctionWrapper{Float64, Tuple{Float64}}(sin),
        FunctionWrapper{Float64, Tuple{Float64}}(x -> x * _data[]),
    ]
        TestUtils.test_tangent_consistency(rng, p)
        TestUtils.test_fwds_rvs_data(rng, p)
    end
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:function_wrappers))
end
