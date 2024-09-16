@testset "luxlib" begin
    @testset "$(typeof(fargs))" for (interface_only, perf_flag, is_primitive, fargs...) in Any[
        (false, :none, true, LuxLib.Impl.matmul, randn(5, 4), randn(4, 3)),
        (false, :none, true, LuxLib.Impl.matmuladd, randn(5, 4), randn(4, 3), randn(5)),
    ]
        test_rule(sr(1), fargs...; perf_flag, is_primitive, interface_only)
    end
end
