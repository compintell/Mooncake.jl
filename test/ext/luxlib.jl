@testset "luxlib" begin
    @testset "$(typeof(fargs))" for (interface_only, perf_flag, is_primitive, fargs...) in vcat(
        Any[
            (false, :none, true, LuxLib.Impl.matmul, randn(5, 4), randn(4, 3)),
            (false, :none, true, LuxLib.Impl.matmuladd, randn(5, 4), randn(4, 3), randn(5)),
            (false, :none, true, LuxLib.Impl.batched_matmul, randn(5, 4, 3), randn(4, 3, 3)),
            (false, :none, false, LuxLib.Impl.activation, Lux.relu, randn(5, 4)),
            (
                false, :none, false,
                LuxLib.Impl.bias_activation_loop!,
                randn(5, 4, 3),
                Lux.relu,
                randn(5, 4, 3),
                randn(4),
            ),
            (
                false, :none, false,
                LuxLib.Impl.activation_loop!, randn(5, 3), NNlib.gelu, randn(5, 3),
            ),
            (false, :stability_and_allocs, true, SLEEFActivations.sigmoid_fast, randn()),
            (false, :stability_and_allocs, true, SLEEFActivations.softplus, randn()),
            (false, :stability_and_allocs, true, SLEEFActivations.logsigmoid, randn()),
            (false, :stability_and_allocs, true, SLEEFActivations.swish, randn()),
            (false, :stability_and_allocs, true, SLEEFActivations.lisht, randn()),
            (false, :stability_and_allocs, true, SLEEFActivations.tanh, randn()),
            (false, :stability_and_allocs, true, SLEEFActivations.tanh_fast, randn()),
            (
                false, :stability_and_allocs, true,
                LuxLib.Utils.static_training_mode_check,
                nothing,
                LuxLib.Utils.True(),
                LuxLib.Utils.True(),
            ),
            (
                false, :none, false,
                function(opmode, act, x, m, sigma2, gamma, beta)
                    LuxLib.Impl.batchnorm_affine_normalize_internal(
                        opmode, act, x, m, sigma2, gamma, beta, 1e-3
                    )
                end,
                LuxLib.LoopedArrayOp(),
                Lux.relu,
                randn(5, 4, 3),
                randn(4),
                rand(4) .+ 1.0,
                nothing,
                nothing,
            ),
        ],
        vec(map(Iterators.product(
            [LuxLib.LoopedArrayOp(), LuxLib.GenericBroadcastOp()],
            [randn(5), nothing],
            [Lux.relu, tanh, NNlib.gelu],
        )) do (opmode, bias, activation)
            (
                false, :none, false,
                LuxLib.Impl.fused_dense, opmode, activation, randn(5, 4), randn(4, 2), bias,
            )
        end),
    )
        test_rule(sr(1), fargs...; perf_flag, is_primitive, interface_only)
    end
end
