@testset "new" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:performance_patches))
end
