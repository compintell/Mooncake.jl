multiarg_fn(x) = only(x)
multiarg_fn(x, y) = only(x) + only(y)
multiarg_fn(x, y, z) = only(x) + only(y) + only(z)

vararg_fn(x) = multiarg_fn(x...)

@testset "umlaut_internals_rules" begin

    @testset "misc utility" begin
        x = randn(4, 5)
        p = Base.unsafe_convert(Ptr{Float64}, x)
        @test Taped.wrap_ptr_as_view(p, 4, 4, 5) == x
        @test Taped.wrap_ptr_as_view(p, 4, 2, 5) == x[1:2, :]
        @test Taped.wrap_ptr_as_view(p, 4, 2, 3) == x[1:2, 1:3]
    end

    TestUtils.run_hand_written_rrule!!_test_cases(StableRNG, Val(:umlaut_internals_rules))

    @testset for (interface_only, f, x...) in Any[
        (false, x -> multiarg_fn(x...), 1),
        (false, x -> multiarg_fn(x...), [1.0, 2.0]),
        (false, x -> multiarg_fn(x...), [5.0, 4]),
        (false, x -> multiarg_fn(x...), (5.0, 4)),
        (false, x -> multiarg_fn(x...), (a=5.0, b=4)),
        (false, x -> multiarg_fn(x...), svec(5.0, 4.0)),
    ]
        test_taped_rrule!!(sr(123), f, map(deepcopy, x)...; interface_only)
    end
end
