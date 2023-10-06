@testset "builtins" begin
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

    @test_throws(
        ErrorException,
        Taped.rrule!!(CoDual(IntrinsicsWrappers.add_ptr, NoTangent()), 5.0, 4.0),
    )

    @test_throws(
        ErrorException,
        Taped.rrule!!(CoDual(IntrinsicsWrappers.sub_ptr, NoTangent()), 5.0, 4.0),
    )

    _x = Ref(5.0) # data used in tests which aren't protected by GC.
    _dx = Ref(4.0)
    @testset "$f, $(typeof(x))" for (interface_only, perf_flag, f, x...) in [

        # IR-node workarounds:
        (false, :none, Taped.Umlaut.__new__, UnitRange{Int}, 5, 9),
        (false, :none, Taped.Umlaut.__new__, TestResources.StructFoo, 5.0, randn(4)),
        (false, :none, Taped.Umlaut.__new__, TestResources.MutableFoo, 5.0, randn(5)),
        (false, :none, Taped.Umlaut.__new__, NamedTuple{(), Tuple{}}),

        # Core.Intrinsics:
        (false, :none, IntrinsicsWrappers.abs_float, 5.0),
        (false, :none, IntrinsicsWrappers.add_float, 4.0, 5.0),
        (false, :none, IntrinsicsWrappers.add_float_fast, 4.0, 5.0),
        (false, :none, IntrinsicsWrappers.add_int, 1, 2),
        (false, :none, IntrinsicsWrappers.and_int, 2, 3),
        (false, :none, IntrinsicsWrappers.arraylen, randn(10)),
        (false, :none, IntrinsicsWrappers.arraylen, randn(10, 7)),
        (false, :none, IntrinsicsWrappers.ashr_int, 123456, 0x0000000000000020),
        # atomic_fence -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointermodify -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerref -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerreplace -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerset -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerswap -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, IntrinsicsWrappers.bitcast, Float64, 5),
        (false, :none, IntrinsicsWrappers.bitcast, Int64, 5.0),
        (false, :none, IntrinsicsWrappers.bswap_int, 5),
        (false, :none, IntrinsicsWrappers.ceil_llvm, 4.1),
        # cglobal -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, IntrinsicsWrappers.checked_sadd_int, 5, 4),
        (false, :none, IntrinsicsWrappers.checked_sdiv_int, 5, 4),
        (false, :none, IntrinsicsWrappers.checked_smul_int, 5, 4),
        (false, :none, IntrinsicsWrappers.checked_srem_int, 5, 4),
        (false, :none, IntrinsicsWrappers.checked_ssub_int, 5, 4),
        (false, :none, IntrinsicsWrappers.checked_uadd_int, 5, 4),
        (false, :none, IntrinsicsWrappers.checked_udiv_int, 5, 4),
        (false, :none, IntrinsicsWrappers.checked_umul_int, 5, 4),
        (false, :none, IntrinsicsWrappers.checked_urem_int, 5, 4),
        (false, :none, IntrinsicsWrappers.checked_usub_int, 5, 4),
        (false, :none, IntrinsicsWrappers.copysign_float, 5.0, 4.0),
        (false, :none, IntrinsicsWrappers.copysign_float, 5.0, -3.0),
        (false, :none, IntrinsicsWrappers.ctlz_int, 5),
        (false, :none, IntrinsicsWrappers.ctpop_int, 5),
        (false, :none, IntrinsicsWrappers.cttz_int, 5),
        (false, :none, IntrinsicsWrappers.div_float, 5.0, 3.0),
        (false, :none, IntrinsicsWrappers.div_float_fast, 5.0, 3.0),
        (false, :none, IntrinsicsWrappers.eq_float, 5.0, 4.0),
        (false, :none, IntrinsicsWrappers.eq_float, 4.0, 4.0),
        (false, :none, IntrinsicsWrappers.eq_float_fast, 5.0, 4.0),
        (false, :none, IntrinsicsWrappers.eq_float_fast, 4.0, 4.0),
        (false, :none, IntrinsicsWrappers.eq_int, 5, 4),
        (false, :none, IntrinsicsWrappers.eq_int, 4, 4),
        (false, :none, IntrinsicsWrappers.flipsign_int, 4, -3),
        (false, :none, IntrinsicsWrappers.floor_llvm, 4.1),
        (false, :none, IntrinsicsWrappers.fma_float, 5.0, 4.0, 3.0),
        # fpext -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, IntrinsicsWrappers.fpiseq, 4.1, 4.0),
        (false, :none, IntrinsicsWrappers.fptosi, UInt32, 4.1),
        (false, :none, IntrinsicsWrappers.fptoui, Int32, 4.1),
        # fptrunc -- maybe interesting
        (true, :none, IntrinsicsWrappers.have_fma, Float64),
        (false, :none, IntrinsicsWrappers.le_float, 4.1, 4.0),
        (false, :none, IntrinsicsWrappers.le_float_fast, 4.1, 4.0),
        # llvm_call -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, IntrinsicsWrappers.lshr_int, 1308622848, 0x0000000000000018),
        (false, :none, IntrinsicsWrappers.lt_float, 4.1, 4.0),
        (false, :none, IntrinsicsWrappers.lt_float_fast, 4.1, 4.0),
        (false, :none, IntrinsicsWrappers.mul_float, 5.0, 4.0),
        (false, :none, IntrinsicsWrappers.mul_float_fast, 5.0, 4.0),
        (false, :none, IntrinsicsWrappers.mul_int, 5, 4),
        (false, :none, IntrinsicsWrappers.muladd_float, 5.0, 4.0, 3.0),
        (false, :none, IntrinsicsWrappers.ne_float, 5.0, 4.0),
        (false, :none, IntrinsicsWrappers.ne_float_fast, 5.0, 4.0),
        (false, :none, IntrinsicsWrappers.ne_int, 5, 4),
        (false, :none, IntrinsicsWrappers.ne_int, 5, 5),
        (false, :none, IntrinsicsWrappers.neg_float, 5.0),
        (false, :none, IntrinsicsWrappers.neg_float_fast, 5.0),
        (false, :none, IntrinsicsWrappers.neg_int, 5),
        (false, :none, IntrinsicsWrappers.not_int, 5),
        (false, :none, IntrinsicsWrappers.or_int, 5, 5),
        # pointerref -- integration tested because pointers are awkward. See below.
        # pointerset -- integration tested because pointers are awkward. See below.
        # rem_float -- untested and unimplemented because seemingly unused on master
        # rem_float_fast -- untested and unimplemented because seemingly unused on master
        (false, :none, IntrinsicsWrappers.rint_llvm, 5),
        (false, :none, IntrinsicsWrappers.sdiv_int, 5, 4),
        (false, :none, IntrinsicsWrappers.sext_int, Int64, Int32(1308622848)),
        (false, :none, IntrinsicsWrappers.shl_int, 1308622848, 0xffffffffffffffe8),
        (false, :none, IntrinsicsWrappers.sitofp, Float64, 0),
        (false, :none, IntrinsicsWrappers.sle_int, 5, 4),
        (false, :none, IntrinsicsWrappers.slt_int, 4, 5),
        (false, :none, IntrinsicsWrappers.sqrt_llvm, 5.0),
        (false, :none, IntrinsicsWrappers.sqrt_llvm_fast, 5.0),
        (false, :none, IntrinsicsWrappers.srem_int, 4, 1),
        (false, :none, IntrinsicsWrappers.sub_float, 4.0, 1.0),
        (false, :none, IntrinsicsWrappers.sub_float_fast, 4.0, 1.0),
        (false, :none, IntrinsicsWrappers.sub_int, 4, 1),
        (false, :none, IntrinsicsWrappers.trunc_int, UInt8, 78),
        (false, :none, IntrinsicsWrappers.trunc_llvm, 5.1),
        (false, :none, IntrinsicsWrappers.udiv_int, 5, 4),
        (false, :none, IntrinsicsWrappers.uitofp, Float16, 4),
        (false, :none, IntrinsicsWrappers.ule_int, 5, 4),
        (false, :none, IntrinsicsWrappers.ult_int, 5, 4),
        (false, :none, IntrinsicsWrappers.urem_int, 5, 4),
        (false, :none, IntrinsicsWrappers.xor_int, 5, 4),
        (false, :none, IntrinsicsWrappers.zext_int, Int64, 0xffffffff),

        # Non-intrinsic built-ins:
        # Core._abstracttype -- NEEDS IMPLEMENTING AND TESTING
        # Core._apply_iterate -- NEEDS IMPLEMENTING AND TESTING
        # Core._apply_pure -- NEEDS IMPLEMENTING AND TESTING
        # Core._call_in_world -- NEEDS IMPLEMENTING AND TESTING
        # Core._call_in_world_total -- NEEDS IMPLEMENTING AND TESTING
        # Core._call_latest -- NEEDS IMPLEMENTING AND TESTING
        # Core._compute_sparams -- NEEDS IMPLEMENTING AND TESTING
        # Core._equiv_typedef -- NEEDS IMPLEMENTING AND TESTING
        # Core._expr -- NEEDS IMPLEMENTING AND TESTING
        # Core._primitivetype -- NEEDS IMPLEMENTING AND TESTING
        # Core._setsuper! -- NEEDS IMPLEMENTING AND TESTING
        # Core._structtype -- NEEDS IMPLEMENTING AND TESTING
        # Core._svec_ref -- NEEDS IMPLEMENTING AND TESTING
        # Core._typebody! -- NEEDS IMPLEMENTING AND TESTING
        (true, :none, Core._typevar, :T, Union{}, Any),
        (false, :none, <:, Float64, Int),
        (false, :none, <:, Any, Float64),
        (false, :none, <:, Float64, Any),
        (false, :none, ===, 5.0, 4.0),
        (false, :none, ===, 5.0, randn(5)),
        (false, :none, ===, randn(5), randn(3)),
        (false, :none, ===, 5.0, 5.0),
        (false, :none, Core.apply_type, Vector, Float64),
        (false, :none, Core.apply_type, Array, Float64, 2),
        (false, :none, Core.arraysize, randn(5, 4, 3), 2),
        (false, :none, Core.arraysize, randn(5, 4, 3, 2, 1), 100),
        # Core.compilerbarrier -- NEEDS IMPLEMENTING AND TESTING
        # Core.const_arrayref -- NEEDS IMPLEMENTING AND TESTING
        # Core.donotdelete -- NEEDS IMPLEMENTING AND TESTING
        # Core.finalizer -- NEEDS IMPLEMENTING AND TESTING
        # Core.get_binding_type -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, Core.ifelse, true, randn(5), 1),
        (false, :none, Core.ifelse, false, randn(5), 2),
        # Core.set_binding_type! -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, Core.sizeof, Float64),
        (false, :none, Core.sizeof, randn(5)),
        # Core.svec -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, Base.arrayref, true, randn(5), 1),
        (false, :none, Base.arrayref, false, randn(4), 1),
        (false, :none, Base.arrayref, true, randn(5, 4), 1, 1),
        (false, :none, Base.arrayref, false, randn(5, 4), 5, 4),
        (false, :none, Base.arrayset, false, randn(5), 4.0, 3),
        (false, :none, Base.arrayset, false, randn(5, 4), 3.0, 1, 3),
        (false, :none, Base.arrayset, true, randn(5), 4.0, 3),
        (false, :none, Base.arrayset, true, randn(5, 4), 3.0, 1, 3),
        (false, :none, applicable, sin, Float64),
        (false, :none, applicable, sin, Type),
        (false, :none, applicable, +, Type, Float64),
        (false, :none, applicable, +, Float64, Float64),
        (false, :none, fieldtype, TestResources.StructFoo, :a),
        (false, :none, fieldtype, TestResources.StructFoo, :b),
        (false, :none, fieldtype, TestResources.MutableFoo, :a),
        (false, :none, fieldtype, TestResources.MutableFoo, :b),
        (true, :none, getfield, TestResources.StructFoo(5.0), :a),
        (false, :none, getfield, TestResources.StructFoo(5.0, randn(5)), :a),
        (false, :none, getfield, TestResources.StructFoo(5.0, randn(5)), :b),
        (true, :none, getfield, TestResources.StructFoo(5.0), 1),
        (false, :none, getfield, TestResources.StructFoo(5.0, randn(5)), 1),
        (false, :none, getfield, TestResources.StructFoo(5.0, randn(5)), 2),
        (true, :none, getfield, TestResources.MutableFoo(5.0), :a),
        (false, :none, getfield, TestResources.MutableFoo(5.0, randn(5)), :b),
        (false, :none, getfield, UnitRange{Int}(5:9), :start),
        (false, :none, getfield, UnitRange{Int}(5:9), :stop),
        (false, :none, getfield, (5.0, ), 1, false),
        (false, :none, getfield, UInt8, :name),
        (false, :none, getfield, UInt8, :super),
        (true, :none, getfield, UInt8, :layout),
        (false, :none, getfield, UInt8, :hash),
        (false, :none, getfield, UInt8, :flags),
        # getglobal requires compositional testing, because you can't deepcopy a module
        # invoke -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, isa, 5.0, Float64),
        (false, :none, isa, 1, Float64),
        (false, :none, isdefined, TestResources.MutableFoo(5.0, randn(5)), :sim),
        (false, :none, isdefined, TestResources.MutableFoo(5.0, randn(5)), :a),
        # modifyfield! -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, nfields, TestResources.MutableFoo),
        (false, :none, nfields, TestResources.StructFoo),
        # replacefield! -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, setfield!, TestResources.MutableFoo(5.0, randn(5)), :a, 4.0),
        (false, :none, setfield!, TestResources.MutableFoo(5.0, randn(5)), :b, randn(5)),
        (false, :none, setfield!, TestResources.MutableFoo(5.0, randn(5)), 1, 4.0),
        (false, :none, setfield!, TestResources.MutableFoo(5.0, randn(5)), 2, randn(5)),
        # swapfield! -- NEEDS IMPLEMENTING AND TESTING
        # throw -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, tuple, 5.0, 4.0),
        (false, :none, tuple, randn(5), 5.0),
        (false, :none, tuple, randn(5), randn(4)),
        (false, :none, tuple, 5.0, randn(1)),
        (false, :none, typeassert, 5.0, Float64),
        (false, :none, typeassert, randn(5), Vector{Float64}),
        (false, :none, typeof, 5.0),
        (false, :none, typeof, randn(5)),
    ]
        test_rrule!!(Xoshiro(123456), f, x...; interface_only, perf_flag)
    end
    @testset for (interface_only, perf_flag, f, x...) in vcat(
        (
            false,
            :none,
            x -> pointerref(bitcast(Ptr{Float64}, pointer_from_objref(Ref(x))), 1, 1),
            5.0,
        ),
        (false, :none, (v, x) -> (pointerset(pointer(x), v, 2, 1); x), 3.0, randn(5)),
        (false, :none, x -> (pointerset(pointer(x), UInt8(3), 2, 1); x), rand(UInt8, 5)),
    )
        test_taped_rrule!!(Xoshiro(123456), f, deepcopy(x)...; interface_only, perf_flag)
    end
end
