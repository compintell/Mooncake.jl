function tester(x, y)
    p = FunctionWrapper{Float64, Tuple{Float64}}(x -> x * y)
    out = 0.0
    for n in 1:1_000
        out += p(x)
    end
    return out
end

function test_existing_wrapper(p, x)
    out = 0.0
    for n in 1:1_000
        out += p(x)
    end
    return out
end

@testset "function_wrappers" begin
    rng = Xoshiro(123)
    p = FunctionWrapper{Float64, Tuple{Float64}}(sin)
    # TestUtils.test_tangent_consistency(rng, p)
    # TestUtils.test_fwds_rvs_data(rng, p)
    test_rule(sr(123456), tester, 5.0, 4.0; is_primitive=false)
end
# TODO:
# 1. Add proper test cases in the function_wrappers file (for both rrules).
# 2. Add performance tests for situations in which you already have the FunctionWrapper.
# 3. Check that repeated calls of a FunctionWrapper are fast once it is constructed.
