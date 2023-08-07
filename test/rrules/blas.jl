@testset "blas" begin
    @testset for (interface_only, f, x...) in [
        (false, BLAS.scal!, 10, 2.4, randn(30), 2),
        (false, BLAS.gemm!, 'N', 'N', randn(), randn(3, 4), randn(4, 5), randn(), randn(3, 5)),
        (false, BLAS.gemm!, 'T', 'N', randn(), randn(3, 4), randn(3, 5), randn(), randn(4, 5)),
        (false, BLAS.gemm!, 'N', 'T', randn(), randn(3, 4), randn(5, 4), randn(), randn(3, 5)),
        (false, BLAS.gemm!, 'T', 'T', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),
        (false, BLAS.gemm!, 'C', 'N', randn(), randn(3, 4), randn(3, 5), randn(), randn(4, 5)),
        (false, BLAS.gemm!, 'N', 'C', randn(), randn(3, 4), randn(5, 4), randn(), randn(3, 5)),
        (false, BLAS.gemm!, 'C', 'C', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),
        (false, BLAS.gemm!, 'C', 'T', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),
        (false, BLAS.gemm!, 'T', 'C', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),
    ]
        test_taped_rrule!!(Xoshiro(123456), f, map(deepcopy, x)...; interface_only)
    end
end
