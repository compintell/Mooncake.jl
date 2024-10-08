@testset "memory" begin
    TestUtils.test_data(sr(123), fill!(Memory{Float64}(undef, 10), 0.0))
end
