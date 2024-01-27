using AbstractGPs, KernelFunctions

@testset "gp" begin
    interp = Taped.TInterp()
    @testset "kernelmatrix_diag $k, $(typeof(x))" for (k, x) in Iterators.product(
        Any[
            SEKernel(), Matern12Kernel(), Matern32Kernel(), Matern52Kernel()
        ],
        Any[
            randn(10), randn(1), ColVecs(randn(2, 11)), RowVecs(randn(9, 4)),
            range(0.0; step=0.1, length=11),
        ]
    )
        f = kernelmatrix
        x = (k, x, x)
        sig = Tuple{typeof(f), map(typeof, x)...}
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp)
        TestUtils.test_rrule!!(
            sr(123456), in_f, f, x...;
            perf_flag=:none, interface_only=true, is_primitive=false,
        )
    end
end
