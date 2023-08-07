@testset "reverse_mode_ad" begin

    @testset "misc utility" begin
        x = randn(4, 5)
        p = Base.unsafe_convert(Ptr{Float64}, x)
        @test Taped.wrap_ptr_as_view(p, 4, 4, 5) == x
        @test Taped.wrap_ptr_as_view(p, 4, 2, 5) == x[1:2, :]
        @test Taped.wrap_ptr_as_view(p, 4, 2, 3) == x[1:2, 1:3]
    end

    _x = Ref(5.0) # data used in tests which aren't protected by GC.
    _dx = Ref(4.0)
    @testset "$f, $(typeof(x))" for (interface_only, f, x...) in [

        # IR-node workarounds:
        (false, Taped.Umlaut.__new__, UnitRange{Int}, 5, 9),
        (false, Taped.Umlaut.__new__, TestResources.StructFoo, 5.0, randn(4)),
        (false, Taped.Umlaut.__new__, TestResources.MutableFoo, 5.0, randn(5)),
        (false, Taped.Umlaut.__new__, NamedTuple{(), Tuple{}}),

        # Umlaut internals -- getindex occassionally gets pushed onto the tape.
        (false, getindex, (5.0, 5.0), 2),
        (false, getindex, (randn(5), 2), 1),
        (false, getindex, (2, randn(5)), 1),

        # Umlaut limitations:
        (false, eltype, randn(5)),
        (false, eltype, transpose(randn(4, 5))),
        (false, Base.promote_op, transpose, Float64),
        (true, String, lazy"hello world"),

    ]
        test_rrule!!(Xoshiro(123456), f, x...; interface_only)
    end
end
