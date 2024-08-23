using DynamicPPL
using DynamicPPL: ADTypes, LogDensityProblemsAD

@testset "TapirDynamicPPLExt" begin
    demo_model = DynamicPPL.TestUtils.DEMO_MODELS[1]
    new_model = demo_model | (s = [1.0, 2.0],)
    f = DynamicPPL.LogDensityFunction(demo_model)
    ad_f_safe = LogDensityProblemsAD.ADgradient(ADTypes.AutoTapir(true), f)
    new_ad_f_safe = DynamicPPL.setmodel(ad_f_safe, new_model, ADTypes.AutoTapir(true))
    @test new_ad_f_safe.ℓ.x.model === new_model
    @test new_ad_f_safe isa LogDensityProblemsAD.ADGradientWrapper
    @test new_ad_f_safe.rule isa Tapir.SafeRRule
    @test new_ad_f_safe.rule === ad_f_safe.rule
    ad_f_unsafe = LogDensityProblemsAD.ADgradient(ADTypes.AutoTapir(false), f)
    new_ad_f_unsafe = DynamicPPL.setmodel(ad_f_unsafe, new_model, ADTypes.AutoTapir(false))
    @test new_ad_f_unsafe.ℓ.x.model === new_model
    @test new_ad_f_unsafe isa LogDensityProblemsAD.ADGradientWrapper
    @test new_ad_f_unsafe.rule isa Tapir.DerivedRule
    @test new_ad_f_unsafe.rule === ad_f_unsafe.rule
end
