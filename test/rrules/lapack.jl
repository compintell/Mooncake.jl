@testset "lapack" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:lapack))
end
