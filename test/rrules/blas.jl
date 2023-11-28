@testset "blas" begin
    t_flags = ['N', 'T', 'C']
    aliased_gemm! = (tA, tB, a, b, A, C) -> BLAS.gemm!(tA, tB, a, A, A, b, C)

    @testset for (interface_only, f, x...) in vcat(

        #
        # BLAS LEVEL 1
        #
        [
            (false, BLAS.dot, 3, randn(5), 1, randn(4), 1),
            (false, BLAS.dot, 3, randn(6), 2, randn(4), 1),
            (false, BLAS.dot, 3, randn(6), 1, randn(9), 3),
            (false, BLAS.dot, 3, randn(12), 3, randn(9), 2),
            (false, BLAS.scal!, 10, 2.4, randn(30), 2),
        ],

        #
        # BLAS LEVEL 2
        #

        # gemv!
        vec(reduce(
            vcat,
            map(product(t_flags, [1, 3], [1, 2])) do (tA, M, N)
                t = tA == 'N'
                As = [
                    t ? randn(M, N) : randn(N, M),
                    view(randn(15, 15), t ? (3:M+2) : (2:N+1), t ? (2:N+1) : (3:M+2)),
                ]
                xs = [randn(N), view(randn(15), 3:N+2), view(randn(30), 1:2:2N)]
                ys = [randn(M), view(randn(15), 2:M+1), view(randn(30), 2:2:2M)]
                return map(Iterators.product(As, xs, ys)) do (A, x, y)
                    (false, BLAS.gemv!, tA, randn(), A, x, randn(), y)
                end
            end,
        )),

        # trmv!
        vec(reduce(
            vcat,
            map(product(['L', 'U'], ['N', 'T', 'C'], ['N', 'U'], [1, 3])) do (ul, tA, dA, N)
                As = [randn(N, N), view(randn(15, 15), 3:N+2, 4:N+3)]
                bs = [randn(N), view(randn(14), 4:N+3)]
                return map(product(As, bs)) do (A, b)
                    (false, BLAS.trmv!, ul, tA, dA, A, b)
                end
            end,
        )),

        #
        # BLAS LEVEL 3
        #

        # gemm!
        vec(map(product(t_flags, t_flags)) do (tA, tB)
            A = tA == 'N' ? randn(3, 4) : randn(4, 3)
            B = tB == 'N' ? randn(4, 5) : randn(5, 4)
            (false, BLAS.gemm!, tA, tB, randn(), A, B, randn(), randn(3, 5))
        end),
        vec(map(product(t_flags, t_flags)) do (tA, tB)
            (false, aliased_gemm!, tA, tB, randn(), randn(), randn(5, 5), randn(5, 5))
        end),

        # trmm!
        vec(reduce(
            vcat,
            map(
                product(['L', 'R'], ['U', 'L'], t_flags, ['N', 'U'], [1, 3], [1, 2]),
            ) do (side, ul, tA, dA, M, N)
                t = tA == 'N'
                R = side == 'L' ? M : N
                As = [randn(R, R), view(randn(15, 15), 3:R+2, 4:R+3)]
                Bs = [randn(M, N), view(randn(15, 15), 2:M+1, 5:N+4)]
                return map(product(As, Bs)) do (A, B)
                    alpha = randn()
                    (false, BLAS.trmm!, side, ul, tA, dA, alpha, A, B)
                end
            end,
        )),

        # trmm!
        vec(reduce(
            vcat,
            map(
                product(['L', 'R'], ['U', 'L'], t_flags, ['N', 'U'], [1, 3], [1, 2]),
            ) do (side, ul, tA, dA, M, N)
                t = tA == 'N'
                R = side == 'L' ? M : N
                As = [randn(R, R) + 5I, view(randn(15, 15), 3:R+2, 4:R+3) + 5I]
                Bs = [randn(M, N), view(randn(15, 15), 2:M+1, 5:N+4)]
                return map(product(As, Bs)) do (A, B)
                    alpha = randn()
                    (false, BLAS.trsm!, side, ul, tA, dA, alpha, A, B)
                end
            end,
        )),
    )
        test_taped_rrule!!(sr(123), f, map(deepcopy, x)...; interface_only, perf_flag=:none)
    end
end
