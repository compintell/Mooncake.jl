@testset "array_legacy" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:array_legacy))
end
