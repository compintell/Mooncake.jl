module Taped

using
    DiffRules,
    LinearAlgebra,
    Random,
    Setfield,
    Umlaut,
    Test

import Umlaut: isprimitive, Frame, Tracer, __foreigncall__

using Base: IEEEFloat, unsafe_convert, unsafe_pointer_to_objref, pointer_from_objref

using Core.Intrinsics:
    not_int, sitofp, sle_int, slt_int, sub_int, add_int, add_float, mul_float, eq_float,
    bitcast, mul_int, and_int, or_int, pointerref, pointerset, sext_int, lshr_int, neg_int,
    shl_int, trunc_int, add_ptr, arraylen, div_float, lt_float, sqrt_llvm, floor_llvm,
    le_float, fptosi, zext_int, eq_int, ashr_int, checked_srem_int, cttz_int, flipsign_int,
    checked_sdiv_int, checked_smul_int

include("tracing.jl")
include("tangents.jl")
include("reverse_mode_ad.jl")
include("testing.jl")

export
    primal,
    shadow,
    randn_tangent,
    increment!!,
    increment_field!!,
    NoTangent,
    Tangent,
    MutableTangent,
    PossiblyUninitTangent,
    set_to_zero!!,
    set_field_to_zero!!,
    tangent_type,
    zero_tangent,
    _scale,
    _add_to_primal,
    _diff,
    _dot

end
