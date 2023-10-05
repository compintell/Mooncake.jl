using LinearAlgebra.LAPACK: getrf!

@testset "lapack" begin
    getrf_wrapper!(x, check) = getrf!(x; check)
    @testset for (interface_only, f, x...) in [
        (false, getrf_wrapper!, randn(5, 5), false),
        (false, getrf_wrapper!, randn(5, 5), true),
        (false, getrf_wrapper!, view(randn(10, 10), 1:5, 1:5), false),
        (false, getrf_wrapper!, view(randn(10, 10), 1:5, 1:5), true),
        (false, getrf_wrapper!, view(randn(10, 10), 2:7, 3:8), false),
        (false, getrf_wrapper!, view(randn(10, 10), 3:8, 2:7), true),
    ]
        test_taped_rrule!!(Xoshiro(123456), f, map(deepcopy, x)...; interface_only)
    end
end
