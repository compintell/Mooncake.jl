@testset "fastmath" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:fastmath))
end
