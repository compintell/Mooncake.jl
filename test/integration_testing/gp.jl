using AbstractGPs, KernelFunctions

@testset "gp" begin
    interp = Tapir.PInterp()
    base_kernels = Any[
        ZeroKernel(),
        ConstantKernel(; c=1.0),
        SEKernel(),
        Matern12Kernel(),
        Matern32Kernel(),
        Matern52Kernel(),
        LinearKernel(),
        PolynomialKernel(; degree=2, c=0.5),
    ]
    simple_xs = Any[
        randn(10),
        randn(1),
        range(0.0; step=0.1, length=11),
        ColVecs(randn(2, 11)),
        RowVecs(randn(9, 4)),
    ]
    d_2_xs = Any[ColVecs(randn(2, 11)), RowVecs(randn(9, 2))]
    @testset "$k, $(typeof(x1))" for (k, x1) in vcat(
        Any[(k, x) for k in base_kernels for x in simple_xs],
        Any[(with_lengthscale(k, 1.1), x) for k in base_kernels for x in simple_xs],
        Any[(with_lengthscale(k, rand(2)), x) for k in base_kernels for x in d_2_xs],
        Any[(k ∘ LinearTransform(randn(2, 2)), x) for k in base_kernels for x in d_2_xs],
        Any[
            (k ∘ LinearTransform(Diagonal(randn(2))), x) for
                k in base_kernels for x in d_2_xs
        ],
    )
        fx = GP(k)(x1, 1.1)
        @testset "$(_typeof(x))" for x in Any[
            (kernelmatrix, k, x1, x1),
            (kernelmatrix_diag, k, x1, x1),
            (kernelmatrix, k, x1),
            (kernelmatrix_diag, k, x1),
            (rand, Xoshiro(123456), fx),
            (logpdf, fx, rand(fx)),
        ]
            @info typeof(x)
            TestUtils.test_derived_rule(
                sr(123456), x...;
                interp, perf_flag=:none, interface_only=false, is_primitive=false,
            )
        end
    end
end
