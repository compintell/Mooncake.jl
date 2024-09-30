@testset "luxlib" begin
    @testset "$(typeof(fargs))" for (interface_only, perf_flag, is_primitive, fargs...) in vcat(
        Any[
            (false, :none, true, LuxLib.Impl.matmul, randn(5, 4), randn(4, 3)),
            (false, :none, true, LuxLib.Impl.matmuladd, randn(5, 4), randn(4, 3), randn(5)),
            (false, :none, false, LuxLib.Impl.activation, Lux.relu, randn(5, 4)),
            (
                false, :none, false,
                LuxLib.Impl.activation_loop!, randn(5, 3), NNlib.gelu, randn(5, 3),
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
