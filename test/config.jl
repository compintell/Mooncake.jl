@testset "config" begin
    @test !Mooncake.DEFAULT_CONFIG.debug_mode
    @test !Mooncake.DEFAULT_CONFIG.silence_debug_messages
end
