@testset "nnlib" begin
    @testset "$(typeof(fargs))" for (perf_flag, fargs...) in Any[
        (:stability, NNlib.upsample_nearest, randn(3), (2,)),
        (:stability, NNlib.upsample_nearest, randn(3, 2), (2, 2)),
        (:stability, NNlib.upsample_nearest, randn(3, 2, 3), (2, 2, 5)),
    ]
        test_rule(sr(1), fargs...; is_primitive=true, perf_flag)
    end
end
