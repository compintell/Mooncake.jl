@testset "special_functions" begin
    @testset for (interface_only, perf_flag, f, x...) in [
        (false, :stability, erfc, 0.1),
        (false, :stability, erfc, 0.0),
        (false, :stability, erfc, -0.5),
        (false, :stability, loggamma, 0.1),
        (false, :stability, loggamma, 0.3),
        (false, :stability, loggamma, 0.6),
    ]
        test_rrule!!(Xoshiro(123456), f, x...; interface_only, perf_flag)
    end
end
