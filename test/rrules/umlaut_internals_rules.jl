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
    @testset "$f, $(typeof(x))" for (interface_only, perf_flag, f, x...) in [

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
            TestResources.TypeStableMutableStruct{Vector{Float64}},
            5.0,
            randn(5),
        ),
        (false, :stability, __new__, NamedTuple{(), Tuple{}}),


        # Umlaut internals -- getindex occassionally gets pushed onto the tape.
        (false, :none, getindex, (5.0, 5.0), 2),
        (false, :none, getindex, (randn(5), 2), 1),
        (false, :none, getindex, (2, randn(5)), 1),

        # Umlaut limitations:
        (false, :none, eltype, randn(5)),
        (false, :none, eltype, transpose(randn(4, 5))),
        (false, :none, Base.promote_op, transpose, Float64),
        (true, :none, String, lazy"hello world"),

    ]
        test_rrule!!(Xoshiro(123456), f, x...; interface_only, perf_flag)
    end
end
