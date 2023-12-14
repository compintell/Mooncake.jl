module Taped

const CC = Core.Compiler

using
    BenchmarkTools,
    DiffRules,
    ExprTools,
    FunctionWrappers,
    InteractiveUtils,
    LinearAlgebra,
    Random,
    Setfield,
    Umlaut

import Umlaut: isprimitive, Frame, Tracer, __foreigncall__, __to_tuple__, __new__

using Base:
    IEEEFloat, unsafe_convert, unsafe_pointer_to_objref, pointer_from_objref, arrayref,
    arrayset
using Base.Experimental: @opaque
using Base.Iterators: product
using Core:
    Intrinsics, bitcast, SimpleVector, svec, ReturnNode, GotoNode, GotoIfNot, PhiNode,
    PiNode, SSAValue, Argument
using Core.Compiler: IRCode
using Core.Intrinsics: pointerref, pointerset
using FunctionWrappers: FunctionWrapper
using LinearAlgebra.BLAS: @blasfunc, BlasInt, trsm!
using LinearAlgebra.LAPACK: getrf!, getrs!, getri!, trtrs!, potrf!, potrs!

# Needs to be defined before various other things.
function _foreigncall_ end
const Tforeigncall = Union{typeof(_foreigncall_), typeof(__foreigncall__)}

include("tracing.jl")
include("acceleration.jl")
include("tangents.jl")
include("reverse_mode_ad.jl")
include("interpreted_function.jl")
include("test_utils.jl")

include(joinpath("rrules", "avoiding_non_differentiable_code.jl"))
include(joinpath("rrules", "blas.jl"))
include(joinpath("rrules", "builtins.jl"))
include(joinpath("rrules", "foreigncall.jl"))
include(joinpath("rrules", "iddict.jl"))
include(joinpath("rrules", "lapack.jl"))
include(joinpath("rrules", "low_level_maths.jl"))
include(joinpath("rrules", "misc.jl"))
include(joinpath("rrules", "new.jl"))
include(joinpath("rrules", "umlaut_internals_rules.jl"))
include(joinpath("rrules", "unrolled_function.jl"))

export
    primal,
    tangent,
    randn_tangent,
    increment!!,
    increment_field!!,
    NoTangent,
    Tangent,
    MutableTangent,
    PossiblyUninitTangent,
    set_to_zero!!,
    tangent_type,
    zero_tangent,
    _scale,
    _add_to_primal,
    _diff,
    _dot,
    zero_codual,
    rrule!!

end
