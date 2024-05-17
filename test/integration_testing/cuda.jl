using CUDA

@testset "cuda" begin
    interp = Tapir.TapirInterpreter()
    TestUtils.test_derived_rule(
        sr(123456), CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}, undef, 256;
        interp, perf_flag=:stability, interface_only=true, is_primitive=true,
    )
end
