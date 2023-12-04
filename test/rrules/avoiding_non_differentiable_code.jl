@testset "avoiding_non_differentiable_code" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:avoiding_non_differentiable_code))
end
