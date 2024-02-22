@testset "special_functions" begin
    @testset for (interface_only, perf_flag, f, x...) in [
        (false, :stability, airyai, 0.1),
        (false, :stability, airyai, 0.0),
        (false, :stability, airyai, -0.5),
        (false, :stability, airyaix, 0.1),
        (false, :stability, airyaix, 0.05),
        (false, :stability, airyaix, 0.9),
        (false, :stability, erfc, 0.1),
        (false, :stability, erfc, 0.0),
        (false, :stability, erfc, -0.5),
    ]
        test_rrule!!(Xoshiro(123456), f, x...; interface_only, perf_flag)
    end
end
