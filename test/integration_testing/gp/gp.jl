using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AbstractGPs, KernelFunctions, Mooncake, Test

@testset "gp" begin
    ks = Any[
        ZeroKernel(),
        ConstantKernel(; c=1.0),
        SEKernel(),
        Matern12Kernel(),
        Matern32Kernel(),
        Matern52Kernel(),
        LinearKernel(),
        PolynomialKernel(; degree=2, c=0.5),
    ]
    xs = Any[
        (randn(10), randn(10)),
        (randn(1), randn(1)),
        (ColVecs(randn(2, 11)), ColVecs(randn(2, 11))),
        (RowVecs(randn(3, 4)), RowVecs(randn(3, 4))),
    ]
    d_2_xs = Any[
        (ColVecs(randn(2, 11)), ColVecs(randn(2, 11))),
        (RowVecs(randn(9, 2)), RowVecs(randn(9, 2))),
    ]
    @testset "$k, $(typeof(x1))" for (k, x1, x2) in vcat(
        Any[(k, x1, x2) for k in ks for (x1, x2) in xs],
        Any[(with_lengthscale(k, 1.1), x1, x2) for k in ks for (x1, x2) in xs],
        Any[(with_lengthscale(k, rand(2)), x1, x2) for k in ks for (x1, x2) in d_2_xs],
        Any[(k ∘ LinearTransform(randn(2, 2)), x1, x2) for k in ks for (x1, x2) in d_2_xs],
        Any[
            (k ∘ LinearTransform(Diagonal(randn(2))), x1, x2) for
                k in ks for (x1, x2) in d_2_xs
        ],
    )
        fx = GP(k)(x1, 1.1)
        @testset "$(_typeof(args))" for args in Any[
            (kernelmatrix, k, x1, x2),
            (kernelmatrix_diag, k, x1, x2),
            (kernelmatrix, k, x1),
            (kernelmatrix_diag, k, x1),
            (fx -> rand(Xoshiro(123456), fx), fx),
            (logpdf, fx, rand(fx)),
        ]
            @info typeof(args)
            test_rule(sr(123456), args...; is_primitive=false, unsafe_perturb=true)
        end
    end
end
