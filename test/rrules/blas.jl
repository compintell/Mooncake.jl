@testset "blas" begin
    @test_throws ErrorException Mooncake.arrayify(5, 4)
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:blas))

    @testset "mixed complex-real" begin
        TestUtils.test_rule(
            StableRNG(123456), x -> sum(complex(x) * x), rand(5, 5); is_primitive=false
        )
    end
end
