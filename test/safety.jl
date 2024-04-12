@testset "safety" begin

    # Forwards-pass tests.
    x = (CoDual(sin, NoTangent()), CoDual(5.0, NoFwdsData()))
    @test_throws(ArgumentError, Tapir.SafeRRule(rrule!!)(x...))
    x = (CoDual(sin, NoFwdsData()), CoDual(5.0, NoFwdsData()))
    @test_throws(
        ArgumentError, Tapir.SafeRRule((x..., ) -> (CoDual(1.0, 0.0), nothing))(x...)
    )

    # Test that bad rdata is caught as a pre-condition.
    y, pb!! = Tapir.SafeRRule(rrule!!)(zero_fwds_codual(sin), zero_fwds_codual(5.0))
    @test_throws(ArgumentError, pb!!(5))

    # Test that bad rdata is caught as a post-condition.
    rule_with_bad_pb(x::CoDual{Float64}) = x, dy -> 5 # returns the wrong type
    y, pb!! = Tapir.SafeRRule(rule_with_bad_pb)(zero_fwds_codual(5.0))
    @test_throws ArgumentError pb!!(1.0)
end
