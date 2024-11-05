@testset "dev_tools" begin
    sig = Tuple{typeof(sin), Float64}
    @test Mooncake.primal_ir(sig) isa CC.IRCode
    @test Mooncake.fwd_ir(sig) isa CC.IRCode
    @test Mooncake.rvs_ir(sig) isa CC.IRCode
end
