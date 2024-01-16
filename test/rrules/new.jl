@testset "new" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:new))
end
