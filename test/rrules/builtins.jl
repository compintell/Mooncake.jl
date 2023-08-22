@testset "builtins" begin
    @test_throws(
        ErrorException,
        Taped.rrule!!(
            CoDual(__intrinsic__, NoTangent()),
            CoDual(Val(Core.Intrinsics.add_ptr), NoTangent()),
            5.0, 4.0,
        ),
    )

    @test_throws(
        ErrorException,
        Taped.rrule!!(
            CoDual(__intrinsic__, NoTangent()),
            CoDual(Val(Core.Intrinsics.sub_ptr), NoTangent()),
            5.0, 4.0,
        ),
    )

    _x = Ref(5.0) # data used in tests which aren't protected by GC.
    _dx = Ref(4.0)
    @testset "$f, $(typeof(x))" for (interface_only, f, x...) in [

        # IR-node workarounds:
        (false, Taped.Umlaut.__new__, UnitRange{Int}, 5, 9),
        (false, Taped.Umlaut.__new__, TestResources.StructFoo, 5.0, randn(4)),
        (false, Taped.Umlaut.__new__, TestResources.MutableFoo, 5.0, randn(5)),
        (false, Taped.Umlaut.__new__, NamedTuple{(), Tuple{}}),

        # Core.Intrinsics:
        (false, __intrinsic__, Val(Intrinsics.abs_float), 5.0),
        (false, __intrinsic__, Val(Intrinsics.add_float), 4.0, 5.0),
        (false, __intrinsic__, Val(Intrinsics.add_float_fast), 4.0, 5.0),
        (false, __intrinsic__, Val(Intrinsics.add_int), 1, 2),
        (false, __intrinsic__, Val(Intrinsics.and_int), 2, 3),
        (false, __intrinsic__, Val(Intrinsics.arraylen), randn(10)),
        (false, __intrinsic__, Val(Intrinsics.arraylen), randn(10, 7)),
        (false, __intrinsic__, Val(Intrinsics.ashr_int), 123456, 0x0000000000000020),
        # atomic_fence -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointermodify -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerref -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerreplace -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerset -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerswap -- NEEDS IMPLEMENTING AND TESTING
        (false, __intrinsic__, Val(Intrinsics.bitcast), Float64, 5),
        (false, __intrinsic__, Val(Intrinsics.bitcast), Int64, 5.0),
        (false, __intrinsic__, Val(Intrinsics.bswap_int), 5),
        (false, __intrinsic__, Val(Intrinsics.ceil_llvm), 4.1),
        # cglobal -- NEEDS IMPLEMENTING AND TESTING
        (false, __intrinsic__, Val(Intrinsics.checked_sadd_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.checked_sdiv_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.checked_smul_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.checked_srem_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.checked_ssub_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.checked_uadd_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.checked_udiv_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.checked_umul_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.checked_urem_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.checked_usub_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.copysign_float), 5.0, 4.0),
        (false, __intrinsic__, Val(Intrinsics.copysign_float), 5.0, -3.0),
        (false, __intrinsic__, Val(Intrinsics.ctlz_int), 5),
        (false, __intrinsic__, Val(Intrinsics.ctpop_int), 5),
        (false, __intrinsic__, Val(Intrinsics.cttz_int), 5),
        (false, __intrinsic__, Val(Intrinsics.div_float), 5.0, 3.0),
        (false, __intrinsic__, Val(Intrinsics.div_float_fast), 5.0, 3.0),
        (false, __intrinsic__, Val(Intrinsics.eq_float), 5.0, 4.0),
        (false, __intrinsic__, Val(Intrinsics.eq_float), 4.0, 4.0),
        (false, __intrinsic__, Val(Intrinsics.eq_float_fast), 5.0, 4.0),
        (false, __intrinsic__, Val(Intrinsics.eq_float_fast), 4.0, 4.0),
        (false, __intrinsic__, Val(Intrinsics.eq_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.eq_int), 4, 4),
        (false, __intrinsic__, Val(Intrinsics.flipsign_int), 4, -3),
        (false, __intrinsic__, Val(Intrinsics.floor_llvm), 4.1),
        (false, __intrinsic__, Val(Intrinsics.fma_float), 5.0, 4.0, 3.0),
        # fpext -- NEEDS IMPLEMENTING AND TESTING
        (false, __intrinsic__, Val(Intrinsics.fpiseq), 4.1, 4.0),
        (false, __intrinsic__, Val(Intrinsics.fptosi), UInt32, 4.1),
        (false, __intrinsic__, Val(Intrinsics.fptoui), Int32, 4.1),
        # fptrunc -- maybe interesting
        (true, __intrinsic__, Val(Intrinsics.have_fma), Float64),
        (false, __intrinsic__, Val(Intrinsics.le_float), 4.1, 4.0),
        (false, __intrinsic__, Val(Intrinsics.le_float_fast), 4.1, 4.0),
        # llvm_call -- NEEDS IMPLEMENTING AND TESTING
        (false, __intrinsic__, Val(Intrinsics.lshr_int), 1308622848, 0x0000000000000018),
        (false, __intrinsic__, Val(Intrinsics.lt_float), 4.1, 4.0),
        (false, __intrinsic__, Val(Intrinsics.lt_float_fast), 4.1, 4.0),
        (false, __intrinsic__, Val(Intrinsics.mul_float), 5.0, 4.0),
        (false, __intrinsic__, Val(Intrinsics.mul_float_fast), 5.0, 4.0),
        (false, __intrinsic__, Val(Intrinsics.mul_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.muladd_float), 5.0, 4.0, 3.0),
        (false, __intrinsic__, Val(Intrinsics.ne_float), 5.0, 4.0),
        (false, __intrinsic__, Val(Intrinsics.ne_float_fast), 5.0, 4.0),
        (false, __intrinsic__, Val(Intrinsics.ne_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.ne_int), 5, 5),
        (false, __intrinsic__, Val(Intrinsics.neg_float), 5.0),
        (false, __intrinsic__, Val(Intrinsics.neg_float_fast), 5.0),
        (false, __intrinsic__, Val(Intrinsics.neg_int), 5),
        (false, __intrinsic__, Val(Intrinsics.not_int), 5),
        (false, __intrinsic__, Val(Intrinsics.or_int), 5, 5),
        # pointerref -- integration tested because pointers are awkward
        # pointerset -- integration tested because pointers are awkward
        # rem_float -- untested and unimplemented because seemingly unused on master
        # rem_float_fast -- untested and unimplemented because seemingly unused on master
        (false, __intrinsic__, Val(Intrinsics.rint_llvm), 5),
        (false, __intrinsic__, Val(Intrinsics.sdiv_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.sext_int), Int64, Int32(1308622848)),
        (false, __intrinsic__, Val(Intrinsics.shl_int), 1308622848, 0xffffffffffffffe8),
        (false, __intrinsic__, Val(Intrinsics.sitofp), Float64, 0),
        (false, __intrinsic__, Val(Intrinsics.sle_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.slt_int), 4, 5),
        (false, __intrinsic__, Val(Intrinsics.sqrt_llvm), 5.0),
        (false, __intrinsic__, Val(Intrinsics.sqrt_llvm_fast), 5.0),
        (false, __intrinsic__, Val(Intrinsics.srem_int), 4, 1),
        (false, __intrinsic__, Val(Intrinsics.sub_float), 4.0, 1.0),
        (false, __intrinsic__, Val(Intrinsics.sub_float_fast), 4.0, 1.0),
        (false, __intrinsic__, Val(Intrinsics.sub_int), 4, 1),
        (false, __intrinsic__, Val(Intrinsics.trunc_int), UInt8, 78),
        (false, __intrinsic__, Val(Intrinsics.trunc_llvm), 5.1),
        (false, __intrinsic__, Val(Intrinsics.udiv_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.uitofp), Float16, 4),
        (false, __intrinsic__, Val(Intrinsics.ule_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.ult_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.urem_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.xor_int), 5, 4),
        (false, __intrinsic__, Val(Intrinsics.zext_int), Int64, 0xffffffff),

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
        (true, Core._typevar, :T, Union{}, Any),
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
        # Core.compilerbarrier -- NEEDS IMPLEMENTING AND TESTING
        # Core.const_arrayref -- NEEDS IMPLEMENTING AND TESTING
        # Core.donotdelete -- NEEDS IMPLEMENTING AND TESTING
        # Core.finalizer -- NEEDS IMPLEMENTING AND TESTING
        # Core.get_binding_type -- NEEDS IMPLEMENTING AND TESTING
        (false, Core.ifelse, true, randn(5), 1),
        (false, Core.ifelse, false, randn(5), 2),
        # Core.set_binding_type! -- NEEDS IMPLEMENTING AND TESTING
        (false, Core.sizeof, Float64),
        (false, Core.sizeof, randn(5)),
        # Core.svec -- NEEDS IMPLEMENTING AND TESTING
        (false, Base.arrayref, true, randn(5), 1),
        (false, Base.arrayref, false, randn(4), 1),
        (false, Base.arrayref, true, randn(5, 4), 1, 1),
        (false, Base.arrayref, false, randn(5, 4), 5, 4),
        (false, Base.arrayset, false, randn(5), 4.0, 3),
        (false, Base.arrayset, false, randn(5, 4), 3.0, 1, 3),
        (false, Base.arrayset, true, randn(5), 4.0, 3),
        (false, Base.arrayset, true, randn(5, 4), 3.0, 1, 3),
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
        (false, getfield, UInt8, :name),
        (false, getfield, UInt8, :super),
        (true, getfield, UInt8, :layout),
        (false, getfield, UInt8, :hash),
        (false, getfield, UInt8, :flags),
        # getglobal requires compositional testing, because you can't deepcopy a module
        # invoke -- NEEDS IMPLEMENTING AND TESTING
        (false, isa, 5.0, Float64),
        (false, isa, 1, Float64),
        (false, isdefined, TestResources.MutableFoo(5.0, randn(5)), :sim),
        (false, isdefined, TestResources.MutableFoo(5.0, randn(5)), :a),
        # modifyfield! -- NEEDS IMPLEMENTING AND TESTING
        (false, nfields, TestResources.MutableFoo),
        (false, nfields, TestResources.StructFoo),
        # replacefield! -- NEEDS IMPLEMENTING AND TESTING
        (false, setfield!, TestResources.MutableFoo(5.0, randn(5)), :a, 4.0),
        (false, setfield!, TestResources.MutableFoo(5.0, randn(5)), :b, randn(5)),
        # swapfield! -- NEEDS IMPLEMENTING AND TESTING
        # throw -- NEEDS IMPLEMENTING AND TESTING
        (false, tuple, 5.0, 4.0),
        (false, tuple, randn(5), 5.0),
        (false, tuple, randn(5), randn(4)),
        (false, tuple, 5.0, randn(1)),
        (false, typeassert, 5.0, Float64),
        (false, typeassert, randn(5), Vector{Float64}),
        (false, typeof, 5.0),
        (false, typeof, randn(5)),
    ]
        test_rrule!!(
            Xoshiro(123456), f, x...;
            interface_only, check_conditional_type_stability=false,
        )
    end
end
