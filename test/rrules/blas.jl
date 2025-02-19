@testset "blas" begin
    @test_throws ErrorException Mooncake.arrayify(5, 4)
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:blas))
end