overlay_tester(x) = 2x
Mooncake.@mooncake_overlay overlay_tester(x) = 3x

@testset "method_overlays" begin
    rule = Mooncake.build_rrule(Tuple{typeof(overlay_tester), Float64})
    @test value_and_gradient!!(rule, overlay_tester, 5.0) == (15.0, (NoTangent(), 3.0))
end
