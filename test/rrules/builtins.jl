@testset "builtins" begin
    @test_throws(
        ErrorException,
        Tapir.rrule!!(CoDual(IntrinsicsWrappers.add_ptr, NoTangent()), 5.0, 4.0),
    )

    @test_throws(
        ErrorException,
        Tapir.rrule!!(CoDual(IntrinsicsWrappers.sub_ptr, NoTangent()), 5.0, 4.0),
    )

    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:builtins))

    @testset "Disable bitcast to differentiable type" begin
        @test_throws(
            ArgumentError,
            rrule!!(zero_fcodual(bitcast), zero_fcodual(Float64), zero_fcodual(5))
        )
    end
end
