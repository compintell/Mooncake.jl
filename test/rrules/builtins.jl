@testset "builtins" begin
    @test_throws(
        ErrorException,
        Phi.rrule!!(CoDual(IntrinsicsWrappers.add_ptr, NoTangent()), 5.0, 4.0),
    )

    @test_throws(
        ErrorException,
        Phi.rrule!!(CoDual(IntrinsicsWrappers.sub_ptr, NoTangent()), 5.0, 4.0),
    )

    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:builtins))
end
