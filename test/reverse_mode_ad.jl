get_address(x) = ismutable(x) ? pointer_from_objref(x) : nothing

function test_rrule!!(rng::AbstractRNG, x...)

    # Set up problem.
    x_copy = (x[1], map(deepcopy, x[2:end])...)
    x_addresses = map(get_address, x)
    x_x̄ = map(x -> CoDual(x, randn_tangent(rng, x)), x)
    y_ȳ, pb!! = Taped.rrule!!(x_x̄...)
    x = map(primal, x_x̄)
    x̄ = map(shadow, x_x̄)

    # Check output and incremented shadow types are correct.
    @test typeof(primal(y_ȳ)) == typeof(x[1](x[2:end]...))
    @test primal(y_ȳ) == x[1](x[2:end]...)
    @test shadow(y_ȳ) isa tangent_type(typeof(primal(y_ȳ)))
    x̄_new = pb!!(shadow(y_ȳ), x̄...)
    @test all(map((a, b) -> typeof(a) == typeof(b), x̄_new, x̄))

    # Check aliasing.
    @test all(map((x̄, x̄_new) -> ismutable(x̄) ? x̄ === x̄_new : true, x̄, x̄_new))

    # Check that inputs have been returned to their original state.
    @test all(map(==, x, x_copy))

    # Check that memory addresses have remained constant.
    new_x_addresses = map(get_address, x)
    @test all(map(==, x_addresses, new_x_addresses))

    # Check that the answers are numerically correct.
    Taped.test_rmad(rng, x...)
end

function test_taped_rrule!!(rng::AbstractRNG, f, x...)
    _, tape = trace(f, map(deepcopy, x)...; ctx=Taped.RMC())
    f_t = Taped.UnrolledFunction(tape)
    test_rrule!!(rng, f_t, f, x...)
end

@testset "reverse_mode_ad" begin
    @testset "$f, $(typeof(x))" for (f, x...) in [

        # IR-node workarounds:
        (Taped.Umlaut.__new__, UnitRange{Int}, 5, 9),
        (Taped.Umlaut.__new__, TestResources.StructFoo, 5.0, randn(4)),
        (Taped.Umlaut.__new__, TestResources.MutableFoo, 5.0, randn(5)),
        (Taped.Umlaut.__new__, NamedTuple{(), Tuple{}}),

        # Intrinsics:
        (Core.Intrinsics.not_int, 4),
        (Core.Intrinsics.sitofp, Float64, 0),
        (Core.Intrinsics.sle_int, 5, 4),
        (Core.Intrinsics.slt_int, 4, 5),
        (Core.Intrinsics.sub_int, 5, 4),
        (Core.Intrinsics.add_int, 1, 2),
        (Core.Intrinsics.add_float, 4.0, 5.0),
        (Core.Intrinsics.mul_float, 5.0, 4.0),
        (Core.Intrinsics.eq_float, 5.0, 4.0),
        (Core.Intrinsics.eq_float, 4.0, 4.0),
        (Core.Intrinsics.mul_int, 2, 3),
        (Core.Intrinsics.and_int, 2, 3),
        (Core.Intrinsics.or_int, 3, 4),
        (Core.Intrinsics.or_int, true, false),

        # Non-intrinsic built-ins:
        (<:, Float64, Int),
        (<:, Any, Float64),
        (<:, Float64, Any),
        (===, 5.0, 4.0),
        (===, 5.0, randn(5)),
        (===, randn(5), randn(3)),
        (===, 5.0, 5.0),
        (Core.apply_type, Vector, Float64),
        (Core.apply_type, Array, Float64, 2),
        (Core.arraysize, randn(5, 4, 3), 2),
        (Core.arraysize, randn(5, 4, 3, 2, 1), 100),
        (Core.ifelse, true, randn(5), 1),
        (Core.ifelse, false, randn(5), 2),
        (Core.sizeof, Float64),
        (Core.sizeof, randn(5)),
        (applicable, sin, Float64),
        (applicable, sin, Type),
        (applicable, +, Type, Float64),
        (applicable, +, Float64, Float64),
        (fieldtype, TestResources.StructFoo, :a),
        (fieldtype, TestResources.StructFoo, :b),
        (fieldtype, TestResources.MutableFoo, :a),
        (fieldtype, TestResources.MutableFoo, :b),
        # (getfield, TestResources.StructFoo(5.0), :a), # testing limitations
        (getfield, TestResources.StructFoo(5.0, randn(5)), :b),
        # (getfield, TestResources.MutableFoo(5.0), :a), # testing limitations
        (getfield, TestResources.MutableFoo(5.0, randn(5)), :b),
        (getfield, UnitRange{Int}(5:9), :start),
        (getfield, UnitRange{Int}(5:9), :stop),
        (getfield, (5.0, ), 1, false),
        (isa, 5.0, Float64),
        (isa, 1, Float64),
        (isdefined, TestResources.MutableFoo(5.0, randn(5)), :sim),
        (isdefined, TestResources.MutableFoo(5.0, randn(5)), :a),
        (nfields, TestResources.MutableFoo),
        (nfields, TestResources.StructFoo),
        (setfield!, TestResources.MutableFoo(5.0, randn(5)), :a, 4.0),
        (setfield!, TestResources.MutableFoo(5.0, randn(5)), :b, randn(5)),
        (tuple, 5.0, 4.0),
        (tuple, randn(5), 5.0),
        (tuple, randn(5), randn(4)),
        (tuple, 5.0, randn(1)),
        (typeassert, 5.0, Float64),
        (typeassert, randn(5), Vector{Float64}),
        (typeof, 5.0),
        (typeof, randn(5)),

        # Rules to avoid foreigncall nodes:
        (BLAS.gemm!, 'N', 'N', randn(), randn(3, 4), randn(4, 5), randn(), randn(3, 5)),
        (BLAS.gemm!, 'T', 'N', randn(), randn(3, 4), randn(3, 5), randn(), randn(4, 5)),
        (BLAS.gemm!, 'N', 'T', randn(), randn(3, 4), randn(5, 4), randn(), randn(3, 5)),
        (BLAS.gemm!, 'T', 'T', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),
        (BLAS.gemm!, 'C', 'N', randn(), randn(3, 4), randn(3, 5), randn(), randn(4, 5)),
        (BLAS.gemm!, 'N', 'C', randn(), randn(3, 4), randn(5, 4), randn(), randn(3, 5)),
        (BLAS.gemm!, 'C', 'C', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),
        (BLAS.gemm!, 'C', 'T', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),
        (BLAS.gemm!, 'T', 'C', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),

        # Umlaut internals.
        (getindex, (5.0, 5.0), 2),
        (getindex, (randn(5), 2), 1),
        (getindex, (2, randn(5)), 1),

        # Umlaut limitations:
        (eltype, randn(5)),
        (eltype, transpose(randn(4, 5))),
        (Base.promote_op, transpose, Float64),

        # Performance-rules:
        (sin, 5.0),
        (cos, 5.0),
        (getindex, randn(5), 4),
        (getindex, randn(5, 4), 1, 3),
        (setindex!, randn(5), 4.0, 3),
        (setindex!, randn(5, 4), 3.0, 1, 3),
    ]
        test_rrule!!(Xoshiro(123456), f, x...)
    end
    @testset for (f, x...) in vcat(
        TestResources.TEST_FUNCTIONS,
        [
            (mul!, randn(3, 5), randn(3, 4), randn(4, 5), randn(), randn()),
            (mul!, randn(3, 5), randn(4, 3)', randn(4, 5), randn(), randn()),
            (mul!, randn(3, 5), randn(3, 4), randn(5, 4)', randn(), randn()),
            (mul!, randn(3, 5), randn(4, 3)', randn(5, 4)', randn(), randn()),
            (mul!, randn(3, 5), transpose(randn(4, 3)), randn(4, 5), randn(), randn()),
            (mul!, randn(3, 5), randn(3, 4), transpose(randn(5, 4)), randn(), randn()),
            (mul!, randn(3, 5), transpose(randn(4, 3)), transpose(randn(5, 4)), randn(), randn()),
            (view, randn(5, 4), 1, 1),
            (view, randn(5, 4), 2:3, 1),
            (view, randn(5, 4), 1, 2:3),
            (view, randn(5, 4), 2:3, 2:4),

            # This needs some work. Might need to implement gemm! a little bit lower (at the ccall)
            # (mul!, randn(3, 5), view(randn(5, 5), 1:3, 1:4), randn(4, 5), randn(), randn()),
        ]
    )
        test_taped_rrule!!(Xoshiro(123456), f, map(deepcopy, x)...)
    end
end
