@testset "fwds_rvs_data" begin
    @testset "$(typeof(p))" for (_, p, _...) in Tapir.tangent_test_cases()
        TestUtils.test_fwds_rvs_data(Xoshiro(123456), p)
    end
end
