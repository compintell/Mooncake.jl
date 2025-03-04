@testset "randn" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:random))
end
