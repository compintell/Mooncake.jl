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

    _x = Ref(5.0) # data used in tests which aren't protected by GC.
    _dx = Ref(4.0)
    @testset "$f, $(typeof(x))" for (interface_only, perf_flag, f, x...) in Any[

        # IR-node workarounds:
        (false, :stability, __new__, UnitRange{Int}, 5, 9),
        (false, :none, __new__, TestResources.StructFoo, 5.0, randn(4)),
        (false, :none, __new__, TestResources.MutableFoo, 5.0, randn(5)),
        (false, :none, __new__, TestResources.StructFoo, 5.0),
        (false, :none, __new__, TestResources.MutableFoo, 5.0),
        (
            false,
            :stability,
            __new__,
            TypeStableMutableStruct{Vector{Float64}},
            5.0,
            randn(5),
        ),
        (
            false,
            :stability,
            __new__,
            TypeStableMutableStruct{Vector{Float64}},
            5.0,
        ),
        (false, :stability, __new__, NamedTuple{(), Tuple{}}),
        (
            false,
            :stability,
            __new__,
            NamedTuple{(:a, :b), Tuple{Float64, Float64}},
            5.0,
            4.0,
        ),
        (false, :stability, __new__, Tuple{Float64, Float64}, 5.0, 4.0),

        # Splatting primitives:
        (false, :stability, __to_tuple__, (5.0, 4)),
        (false, :stability, __to_tuple__, (a=5.0, b=4)),
        (false, :stability, __to_tuple__, 5),
        (false, :none, __to_tuple__, svec(5.0)),
        (false, :none, __to_tuple__, [5.0, 4.0]),

        # Umlaut limitations:
        (false, :none, eltype, randn(5)),
        (false, :none, eltype, transpose(randn(4, 5))),
        (false, :none, Base.promote_op, transpose, Float64),
        (true, :none, String, lazy"hello world"),
    ]
        test_rrule!!(Xoshiro(123456), f, x...; interface_only, perf_flag)
    end
    @testset for (interface_only, f, x...) in Any[
        (false, x -> multiarg_fn(x...), 1),
        (false, x -> multiarg_fn(x...), [1.0, 2.0]),
        (false, x -> multiarg_fn(x...), [5.0, 4]),
        (false, x -> multiarg_fn(x...), (5.0, 4)),
        (false, x -> multiarg_fn(x...), (a=5.0, b=4)),
        (false, x -> multiarg_fn(x...), svec(5.0, 4.0)),
    ]
        test_taped_rrule!!(Xoshiro(123456), f, map(deepcopy, x)...; interface_only)
    end
end
