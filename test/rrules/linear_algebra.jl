@testset "linear_algebra" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:linear_algebra))
end