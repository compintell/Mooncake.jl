@testset "builtins" begin
    @test_throws(
        ErrorException,
        Taped.rrule!!(CoDual(IntrinsicsWrappers.add_ptr, NoTangent()), 5.0, 4.0),
    )

    @test_throws(
        ErrorException,
        Taped.rrule!!(CoDual(IntrinsicsWrappers.sub_ptr, NoTangent()), 5.0, 4.0),
    )

    TestUtils.run_hand_written_rrule!!_test_cases(StableRNG, Val(:builtins))

    @testset for (interface_only, perf_flag, f, x...) in vcat(
        (
            false,
            :none,
            x -> pointerref(bitcast(Ptr{Float64}, pointer_from_objref(Ref(x))), 1, 1),
            5.0,
        ),
        (false, :none, (v, x) -> (pointerset(pointer(x), v, 2, 1); x), 3.0, randn(5)),
        (false, :none, x -> (pointerset(pointer(x), UInt8(3), 2, 1); x), rand(UInt8, 5)),
        (false, :none, getindex, randn(5), [1, 1]),
        (false, :none, getindex, randn(5), [1, 2, 2]),
        (false, :none, setindex!, randn(5), [4.0, 5.0], [1, 1]),
        (false, :none, setindex!, randn(5), [4.0, 5.0, 6.0], [1, 2, 2]),
    )
        test_taped_rrule!!(sr(123), f, deepcopy(x)...; interface_only, perf_flag)
    end
end
