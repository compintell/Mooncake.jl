using ADTypes, LogDensityProblemsAD
using LogDensityProblemsAD: logdensity_and_gradient, capabilities, dimension, logdensity

# Copied over from LogDensityProblemsAD test suite.
struct TestLogDensity2 end
LogDensityProblemsAD.logdensity(::TestLogDensity2, x) = -sum(abs2, x)
LogDensityProblemsAD.dimension(::TestLogDensity2) = 20
test_gradient(x) = -2 .* x

@testset "AD via Mooncake" begin
    l = TestLogDensity2()
    ∇l = ADgradient(Val(:Mooncake), l)

    @test dimension(∇l) == 20
    @test capabilities(∇l) == LogDensityProblemsAD.LogDensityOrder(1)
    for _ in 1:100
        x = randn(20)
        @test isapprox(@inferred(logdensity(∇l, x)), logdensity(l, x))
        @test isapprox(logdensity_and_gradient(∇l, x)[1], logdensity(TestLogDensity2(), x))
        @test isapprox(logdensity_and_gradient(∇l, x)[2], test_gradient(x))
    end

    config = Mooncake.Config(; debug_mode=false)
    @test ADgradient(ADTypes.AutoMooncake(; config), l) isa typeof(∇l)
    @test parent(∇l) === l

    # Check that nothing for config works.
    @test ADgradient(ADTypes.AutoMooncake(; config=nothing), l) isa typeof(∇l)
    @test parent(∇l) === l

    # Run in debug mode.
    debug_config = Mooncake.Config(; debug_mode=true)
    @test parent(ADgradient(ADTypes.AutoMooncake(; config=debug_config), l)) === l
end
