get_address(x) = ismutable(x) ? pointer_from_objref(x) : nothing

apply(f, x...) = f(x...)

function test_rrule!!(rng::AbstractRNG, x...; interface_only=false, is_primitive=true)

    # Set up problem.
    x_copy = (x[1], map(deepcopy, x[2:end])...)
    x_addresses = map(get_address, x)
    x_x̄ = map(x -> x isa CoDual ? x : CoDual(x, randn_tangent(rng, x)), x)

    # Check that input types are valid.
    for x_x̄ in x_x̄
        @test typeof(shadow(x_x̄)) == tangent_type(typeof(primal(x_x̄)))
    end

    # Attempt to run primal programme. Throw the original exception and provide a little
    # additional context
    x_p = map(primal, x_x̄)
    x_p = (x_p[1], map(deepcopy, x_p[2:end])...)
    try
        apply(x_p...)
    catch e
        display(e)
        println()
        throw(ArgumentError("Primal evaluation does not work."))
    end

    # Verify that the function to which the rrule applies is considered a primitive.
    is_primitive && @test Umlaut.isprimitive(Taped.RMC(), x_p...)

    # Run the rrule and extract results.
    y_ȳ, pb!! = Taped.rrule!!(x_x̄...)
    x = map(primal, x_x̄)
    x̄ = map(shadow, x_x̄)

    # Check output and incremented shadow types are correct.
    @test y_ȳ isa CoDual
    @test typeof(primal(y_ȳ)) == typeof(x[1](x[2:end]...))
    !interface_only && @test primal(y_ȳ) == x[1](x[2:end]...)
    @test shadow(y_ȳ) isa tangent_type(typeof(primal(y_ȳ)))
    x̄_new = pb!!(shadow(y_ȳ), x̄...)
    @test all(map((a, b) -> typeof(a) == typeof(b), x̄_new, x̄))

    # Check aliasing.
    @test all(map((x̄, x̄_new) -> ismutable(x̄) ? x̄ === x̄_new : true, x̄, x̄_new))

    # Check that inputs have been returned to their original state.
    !interface_only && @test all(map(==, x, x_copy))

    # Check that memory addresses have remained constant.
    new_x_addresses = map(get_address, x)
    @test all(map(==, x_addresses, new_x_addresses))

    # Check that the answers are numerically correct.
    !interface_only && Taped.test_rmad(rng, x...)
end

function test_taped_rrule!!(rng::AbstractRNG, f, x...; interface_only=false)
    _, tape = trace(f, map(deepcopy, x)...; ctx=Taped.RMC())
    f_t = Taped.UnrolledFunction(tape)
    test_rrule!!(rng, f_t, f, x...; interface_only, is_primitive=false)
end

# Health Warning: some of these tests are "interface-only" tests. They enable the
# interface_only flag in `test_rrule!!`, and only check that input / output types are
# correct, without tests numerics. This is sometimes necessary, as it's not always possible
# to construct a finite-differencing test to compare against. In such cases, it is strongly
# advised to include a tests which make use of the function in question in composition
# with other functions. `Ptr`s usually require this kind of treatment.
@testset "reverse_mode_ad" begin

    @testset "misc utility" begin
        x = randn(4, 5)
        p = Base.unsafe_convert(Ptr{Float64}, x)
        @test Taped.wrap_ptr_as_view(p, 4, 4, 5) == x
        @test Taped.wrap_ptr_as_view(p, 4, 2, 5) == x[1:2, :]
        @test Taped.wrap_ptr_as_view(p, 4, 2, 3) == x[1:2, 1:3]
    end

    @testset "foreigncalls that should never be hit: $name" for name in [
        :jl_alloc_array_1d, :jl_alloc_array_2d, :jl_alloc_array_3d, :jl_new_array,
        :jl_array_grow_end, :jl_array_del_end, :jl_array_copy, :jl_type_intersection,
        :memset,
    ]
        @test_throws(
            ErrorException,
            Taped.rrule!!(
                CoDual(Umlaut.__foreigncall__, NoTangent()),
                CoDual(Val(name), NoTangent()),
            )
        )
    end

    _x = Ref(5.0) # data used in tests which aren't protected by GC.
    _dx = Ref(4.0)
    @testset "$f, $(typeof(x))" for (interface_only, f, x...) in [

        # IR-node workarounds:
        (false, Taped.Umlaut.__new__, UnitRange{Int}, 5, 9),
        (false, Taped.Umlaut.__new__, TestResources.StructFoo, 5.0, randn(4)),
        (false, Taped.Umlaut.__new__, TestResources.MutableFoo, 5.0, randn(5)),
        (false, Taped.Umlaut.__new__, NamedTuple{(), Tuple{}}),

        # Intrinsics (unary)
        (false, Core.Intrinsics.not_int, 4),
        (false, Core.Intrinsics.neg_int, 5),
        (false, Core.Intrinsics.arraylen, randn(10)),
        (false, Core.Intrinsics.arraylen, randn(10, 7)),
        (false, Core.Intrinsics.sqrt_llvm, 5.0),
        (false, Core.Intrinsics.floor_llvm, 4.3),
        (false, Core.Intrinsics.cttz_int, 4),

        # Intrinsics (binary)
        (false, Core.Intrinsics.sitofp, Float64, 0),
        (false, Core.Intrinsics.sle_int, 5, 4),
        (false, Core.Intrinsics.slt_int, 4, 5),
        (false, Core.Intrinsics.sub_int, 5, 4),
        (false, Core.Intrinsics.add_int, 1, 2),
        (false, Core.Intrinsics.add_float, 4.0, 5.0),
        (false, Core.Intrinsics.mul_float, 5.0, 4.0),
        (false, Core.Intrinsics.eq_float, 5.0, 4.0),
        (false, Core.Intrinsics.eq_float, 4.0, 4.0),
        (false, Core.Intrinsics.bitcast, Float64, 5),
        (false, Core.Intrinsics.bitcast, Int64, 5.0),
        (false, Core.Intrinsics.mul_int, 2, 3),
        (false, Core.Intrinsics.and_int, 2, 3),
        (false, Core.Intrinsics.or_int, 3, 4),
        (false, Core.Intrinsics.or_int, true, false),
        (false, Core.Intrinsics.sext_int, Int64, Int32(1308622848)),
        (false, Core.Intrinsics.lshr_int, 1308622848, 0x0000000000000018),
        (false, Core.Intrinsics.shl_int, 1308622848, 0xffffffffffffffe8),
        (false, Core.Intrinsics.trunc_int, UInt8, 78),
        (false, Core.Intrinsics.div_float, 5.0, 4.0),
        (false, Core.Intrinsics.lt_float, 5.0, 4.0),
        (false, Core.Intrinsics.le_float, 4.0, 3.0),
        (false, Core.Intrinsics.zext_int, Int64, 0xffffffff),
        (false, Core.Intrinsics.eq_int, 5, 4),
        (false, Core.Intrinsics.ashr_int, 123456, 0x0000000000000020),
        (false, Core.Intrinsics.checked_srem_int, 4, 1),
        (false, Core.Intrinsics.flipsign_int, 1, 1),
        (false, Core.Intrinsics.checked_sdiv_int, 4, 1),
        (false, Core.Intrinsics.checked_smul_int, 1, 4),

        # Intrinsics (ternary):
        (
            true,
            pointerref,
            CoDual(
                pointer_from_objref(_x),
                bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(_dx)),
            ),
            1,
            1,
        ),

        # Non-intrinsic built-ins:
        (false, <:, Float64, Int),
        (false, <:, Any, Float64),
        (false, <:, Float64, Any),
        (false, ===, 5.0, 4.0),
        (false, ===, 5.0, randn(5)),
        (false, ===, randn(5), randn(3)),
        (false, ===, 5.0, 5.0),
        (false, Core.apply_type, Vector, Float64),
        (false, Core.apply_type, Array, Float64, 2),
        (false, Core.arraysize, randn(5, 4, 3), 2),
        (false, Core.arraysize, randn(5, 4, 3, 2, 1), 100),
        (false, Core.ifelse, true, randn(5), 1),
        (false, Core.ifelse, false, randn(5), 2),
        (false, Core.sizeof, Float64),
        (false, Core.sizeof, randn(5)),
        (false, applicable, sin, Float64),
        (false, applicable, sin, Type),
        (false, applicable, +, Type, Float64),
        (false, applicable, +, Float64, Float64),
        (false, fieldtype, TestResources.StructFoo, :a),
        (false, fieldtype, TestResources.StructFoo, :b),
        (false, fieldtype, TestResources.MutableFoo, :a),
        (false, fieldtype, TestResources.MutableFoo, :b),
        (true, getfield, TestResources.StructFoo(5.0), :a),
        (false, getfield, TestResources.StructFoo(5.0, randn(5)), :b),
        (true, getfield, TestResources.MutableFoo(5.0), :a),
        (false, getfield, TestResources.MutableFoo(5.0, randn(5)), :b),
        (false, getfield, UnitRange{Int}(5:9), :start),
        (false, getfield, UnitRange{Int}(5:9), :stop),
        (false, getfield, (5.0, ), 1, false),
        (false, getfield, UInt8, :types), # 
        # getglobal requires compositional testing, because you can't deepcopy a module
        (false, isa, 5.0, Float64),
        (false, isa, 1, Float64),
        (false, isdefined, TestResources.MutableFoo(5.0, randn(5)), :sim),
        (false, isdefined, TestResources.MutableFoo(5.0, randn(5)), :a),
        (false, nfields, TestResources.MutableFoo),
        (false, nfields, TestResources.StructFoo),
        (false, setfield!, TestResources.MutableFoo(5.0, randn(5)), :a, 4.0),
        (false, setfield!, TestResources.MutableFoo(5.0, randn(5)), :b, randn(5)),
        (false, tuple, 5.0, 4.0),
        (false, tuple, randn(5), 5.0),
        (false, tuple, randn(5), randn(4)),
        (false, tuple, 5.0, randn(1)),
        (false, typeassert, 5.0, Float64),
        (false, typeassert, randn(5), Vector{Float64}),
        (false, typeof, 5.0),
        (false, typeof, randn(5)),

        # Rules to avoid foreigncall nodes:
        (false, Base.allocatedinline, Float64),
        (false, Base.allocatedinline, Vector{Float64}),
        # (true, pointer_from_objref, _x),
        # (
        #     true,
        #     unsafe_pointer_to_objref,
        #     CoDual(
        #         pointer_from_objref(_x),
        #         bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(_dx)),
        #     ),
        # ),
        (true, Array{Float64, 1}, undef, 5),
        (true, Array{Float64, 2}, undef, 5, 4),
        (true, Array{Float64, 3}, undef, 5, 4, 3),
        (true, Array{Float64, 4}, undef, 5, 4, 3, 2),
        (true, Array{Float64, 5}, undef, 5, 4, 3, 2, 1),
        (true, Array{Float64, 4}, undef, (2, 3, 4, 5)),
        (true, Array{Float64, 5}, undef, (2, 3, 4, 5, 6)),
        (true, Base._growend!, randn(5), 3),
        (false, copy, randn(5, 4)),
        (false, typeintersect, Float64, Int),
        (false, fill!, rand(Int8, 5), Int8(2)),
        (false, fill!, rand(UInt8, 5), UInt8(2)),

        # Umlaut internals.
        (false, getindex, (5.0, 5.0), 2),
        (false, getindex, (randn(5), 2), 1),
        (false, getindex, (2, randn(5)), 1),

        # Umlaut limitations:
        (false, eltype, randn(5)),
        (false, eltype, transpose(randn(4, 5))),
        (false, Base.promote_op, transpose, Float64),
        (true, String, lazy"hello world"),

        # Legit performance rules:
        (false, sin, 5.0),
        (false, cos, 5.0),
        (false, getindex, randn(5), 4),
        (false, getindex, randn(5, 4), 1, 3),
        (false, setindex!, randn(5), 4.0, 3),
        (false, setindex!, randn(5, 4), 3.0, 1, 3),

        # Lack of activity-analysis rules:
        (false, Base.elsize, randn(5, 4)),
        (false, Base.elsize, view(randn(5, 4), 1:2, 1:2)),
        (false, Core.Compiler.sizeof_nothrow, Float64),
        (false, Base.datatype_haspadding, Float64),

        # Rules to avoid pointer type conversions.
        (
            true,
            +,
            CoDual(
                bitcast(Ptr{Float64}, pointer_from_objref(_x)),
                bitcast(Ptr{Float64}, pointer_from_objref(_dx)),
            ),    
            2,
        ),

        # Performance-rules that would ideally be completely removed.
        (false, size, randn(5, 4)),
        (false, LinearAlgebra.lapack_size, 'N', randn(5, 4)),
        (false, Base.require_one_based_indexing, randn(2, 3), randn(2, 1)),
        (false, in, 5.0, randn(4)),
        (false, iszero, 5.0),
        (false, isempty, randn(5)),
        (false, isbitstype, Float64),
        (false, sizeof, Float64),
        (false, promote_type, Float64, Float64),
    ]
        test_rrule!!(Xoshiro(123456), f, x...; interface_only)
    end
    @testset for (interface_only, f, x...) in vcat(
        # TestResources.TEST_FUNCTIONS,
        [
            # (false, x -> getglobal(Main, :sin)(x), 5.0),
            # (false, x -> pointerref(bitcast(Ptr{Float64}, pointer_from_objref(Ref(x))), 1, 1), 5.0),
            # (false, x -> Ref(x)[], 5.0),
            # (false, x -> unsafe_load(bitcast(Ptr{Float64}, pointer_from_objref(Ref(x)))), 5.0),
            # (false, x -> unsafe_load(Base.unsafe_convert(Ptr{Float64}, x)), randn(5)),
            # (false, BLAS.scal!, 10, 2.4, randn(30), 2),
            # # (false, BLAS.scal!, 10, 2f1, randn(Float32, 30), 2),
            # (false, BLAS.gemm!, 'N', 'N', randn(), randn(3, 4), randn(4, 5), randn(), randn(3, 5)),
            # (false, BLAS.gemm!, 'T', 'N', randn(), randn(3, 4), randn(3, 5), randn(), randn(4, 5)),
            # (false, BLAS.gemm!, 'N', 'T', randn(), randn(3, 4), randn(5, 4), randn(), randn(3, 5)),
            # (false, BLAS.gemm!, 'T', 'T', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),
            # (false, BLAS.gemm!, 'C', 'N', randn(), randn(3, 4), randn(3, 5), randn(), randn(4, 5)),
            # (false, BLAS.gemm!, 'N', 'C', randn(), randn(3, 4), randn(5, 4), randn(), randn(3, 5)),
            # (false, BLAS.gemm!, 'C', 'C', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),
            # (false, BLAS.gemm!, 'C', 'T', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),
            # (false, BLAS.gemm!, 'T', 'C', randn(), randn(4, 3), randn(5, 4), randn(), randn(3, 5)),
            # (false, view, randn(5, 4), 1, 1),
            # (false, view, randn(5, 4), 2:3, 1),
            # (false, view, randn(5, 4), 1, 2:3),
            # (false, view, randn(5, 4), 2:3, 2:4),
            # (true, Array{Float64, 1}, undef, (1, )),
            # (true, Array{Float64, 2}, undef, (2, 3)),
            # (true, Array{Float64, 3}, undef, (2, 3, 4)),
            # (true, Xoshiro, 123456),
            # (false, push!, randn(5), 3.0),
        ],
        # vec(map(Iterators.product( # next thing to do is to get all of these working nicely
        #     [randn(3, 5), transpose(randn(5, 3)), adjoint(randn(5, 3))],
        #     [
        #         randn(3, 4),
        #         transpose(randn(4, 3)),
        #         adjoint(randn(4, 3)),
        #         view(randn(5, 5), 1:3, 1:4),
        #         transpose(view(randn(5, 5), 1:4, 1:3)),
        #         adjoint(view(randn(5, 5), 1:4, 1:3)),
        #     ],
        #     [
        #         randn(4, 5),
        #         transpose(randn(5, 4)),
        #         adjoint(randn(5, 4)),
        #         view(randn(5, 5), 1:4, 1:5),
        #         transpose(view(randn(5, 5), 1:5, 1:4)),
        #         adjoint(view(randn(5, 5), 1:5, 1:4)),
        #     ],
        # )) do (A, B, C)
        #     (false, mul!, A, B, C, randn(), randn())
        # end),
    )
        test_taped_rrule!!(Xoshiro(123456), f, map(deepcopy, x)...; interface_only)
    end
end
