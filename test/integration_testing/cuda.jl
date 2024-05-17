using CUDA

@testset "cuda" begin
    TestUtils.test_derived_rule(
        sr(123456), CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}, Undef, 256;
        Tapir.PInterp(), perf_flag=:stability, interface_only=true, is_primitive=true,
    )
end
