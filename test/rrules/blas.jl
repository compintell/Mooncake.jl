@testset "blas" begin
    @test_throws ErrorException Mooncake.arrayify(5, 4)
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:blas))
end

@testset "norm bug" begin
    v = randn(10)
    backend = DifferentiationInterface.AutoMooncake(; config=nothing)
    prep = DifferentiationInterface.prepare_gradient(norm, backend, v)
    g = DifferentiationInterface.gradient(norm, prep, backend, v) # pass
    @test g ≈ v / norm(v)

    v = randn(100)
    prep = DifferentiationInterface.prepare_gradient(norm, backend, v)
    g = DifferentiationInterface.gradient(norm, prep, backend, v) # throw the following error
    @test g ≈ v / norm(v)

    # Real
    v = randn(100)
    Mooncake.TestUtils.test_rule(Random.Xoshiro(123), v->BLAS.nrm2(3, v, 2), v; is_primitive=false)
    Mooncake.TestUtils.test_rule(Random.Xoshiro(123), BLAS.nrm2, v; is_primitive=true)

    # Complex
    v = randn(ComplexF64, 100)
    Mooncake.TestUtils.test_rule(Random.Xoshiro(123), v->BLAS.nrm2(3, v, 2), v; is_primitive=false)
    Mooncake.TestUtils.test_rule(Random.Xoshiro(123), BLAS.nrm2, v; is_primitive=true)
end