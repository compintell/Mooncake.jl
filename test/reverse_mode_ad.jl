@testset "reverse_mode_ad" begin
    @testset "rebind $(typeof(x))" for x in [
        5,
        5.0,
        randn(5),
        (5.0, randn(4), 3),
        (a=5.0, b=3, c=randn(2)),
        TestResources.StructFoo(5.0, randn(5)),
        TestResources.MutableFoo(5.0, randn(5)),
    ]
        test_rrule!!(Xoshiro(123456), rebind, x; interface_only=false, perf_flag=:none)
    end
    @testset "Deduplicated" for (T, unique_args, target_output) in Any[
        Any[Tuple{1, 2, 2}, (+, 5.0), 5.0 + 5.0],
        Any[Tuple{1, 2, 3}, (+, 4.0, 3.0), 4.0 + 3.0],
        Any[Tuple{1, 3, 2}, (+, 4.0, 3.0), 4.0 + 3.0],
        Any[Tuple{1, 2, 3}, (/, 5.0, 2.0), 5.0 / 2.0],
        Any[Tuple{1, 3, 2}, (/, 2.0, 5.0), 5.0 / 2.0],
        Any[
            Tuple{1, 4, 2, 3},
            ((x, y, z) -> x / (sin(y) + z), 2.0, 3.0, 4.0),
            4.0 / (sin(2.0) + 3.0),
        ],
    ]
        @test Deduplicated{T}()(unique_args...) == target_output
        sig = Tuple{Deduplicated{T}, map(Core.Typeof, unique_args)...}
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig)
        TestUtils.test_rrule!!(
            Xoshiro(123456), in_f, unique_args...;
            perf_flag=:none, interface_only=false, is_primitive=false,
        )
    end
end
