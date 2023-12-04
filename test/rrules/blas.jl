@testset "blas" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:blas))
end
