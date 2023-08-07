@testset "blas" begin
    aliased_gemm! = (tA, tB, a, b, A, C) -> BLAS.gemm!(tA, tB, a, A, A, b, C)
    @testset for (interface_only, f, x...) in vcat(
        [
            (false, BLAS.scal!, 10, 2.4, randn(30), 2),
        ],
        vec(map(Iterators.product(['N', 'T', 'C'], ['N', 'T', 'C'])) do (tA, tB)
            A = tA == 'N' ? randn(3, 4) : randn(4, 3)
            B = tB == 'N' ? randn(4, 5) : randn(5, 4)
            (false, BLAS.gemm!, tA, tB, randn(), A, B, randn(), randn(3, 5))
        end),
        vec(map(Iterators.product(['N', 'T', 'C'], ['N', 'T', 'C'])) do (tA, tB)
            (false, aliased_gemm!, tA, tB, randn(), randn(), randn(5, 5), randn(5, 5))
        end),
    )
        test_taped_rrule!!(Xoshiro(123456), f, map(deepcopy, x)...; interface_only)
    end
end
