using LinearAlgebra.LAPACK: getrf!, getrs!, getri!, trtrs!, potrf!

@testset "lapack" begin
    getrf_wrapper!(x, check) = getrf!(x; check)
    @testset for (interface_only, f, x...) in vcat(

        # getrf!
        [
            (false, getrf_wrapper!, randn(5, 5), false),
            (false, getrf_wrapper!, randn(5, 5), true),
            (false, getrf_wrapper!, view(randn(10, 10), 1:5, 1:5), false),
            (false, getrf_wrapper!, view(randn(10, 10), 1:5, 1:5), true),
            (false, getrf_wrapper!, view(randn(10, 10), 2:7, 3:8), false),
            (false, getrf_wrapper!, view(randn(10, 10), 3:8, 2:7), true),
        ],

        # trtrs
        vec(reduce(
            vcat,
            map(product(
                ['U', 'L'], ['N', 'T', 'C'], ['N', 'U'], [1, 3], [1, 2])
            ) do (ul, tA, diag, N, Nrhs)
                As = [randn(N, N) + 10I, view(randn(15, 15) + 10I, 2:N+1, 2:N+1)]
                Bs = [randn(N, Nrhs), view(randn(15, 15), 4:N+3, 3:N+2)]
                return map(product(As, Bs)) do (A, B)
                    (false, trtrs!, ul, tA, diag, A, B)
                end
            end,
        )),

        # getrs
        vec(reduce(
            vcat,
            map(product(['N', 'T'], [1, 9], [1, 2])) do (trans, N, Nrhs)
                As = getrf!.([
                    randn(N, N) + 5I,
                    view(randn(15, 15) + 5I, 2:N+1, 2:N+1),
                ])
                Bs = [randn(N, Nrhs), view(randn(15, 15), 4:N+3, 3:Nrhs+2)]
                return map(product(As, Bs)) do ((A, ipiv), B)
                    (false, getrs!, trans, A, ipiv, B)
                end
            end,
        )),

        # getri
        vec(reduce(
            vcat,
            map([1, 9]) do N
                As = getrf!.([randn(N, N) + 5I, view(randn(15, 15) + I, 2:N+1, 2:N+1)])
                As = getrf!.([randn(N, N) + 5I])
                return map(As) do (A, ipiv)
                    (false, getri!, A, ipiv)
                end
            end,
        )),

        # potrf
        vec(reduce(
            vcat,
            map([1, 3, 9]) do N
                X = randn(N, N)
                A = X * X' + I
                As = [A]
                return map(As) do (A)
                    (false, potrf!, 'L', A)
                end
            end,
        )),
    )
        test_taped_rrule!!(Xoshiro(123456), f, map(deepcopy, x)...; interface_only)
    end
end
