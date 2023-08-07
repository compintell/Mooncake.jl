
#
# Core.Builtin -- these are "primitive" functions which must have rrules because no IR
# is available.
#
# There is a finite number of these functions.
# Any built-ins which don't have rules defined are left as comments with their names
# in this block of code
# As of version 1.9.2 of Julia, there are exactly 139 examples of `Core.Builtin`s.
#

# Note: performance is not considered _at_ _all_ in this implementation.
function rrule!!(f::CoDual{<:Core.IntrinsicFunction}, args...)
    y, pb!! = rrule!!(
        CoDual(__intrinsic__, NoTangent()),
        CoDual(Val(primal(f)), NoTangent()),
        args...
    )
    function intrinsic_function_pullback!!(dy, df, dargs...)
        df, _, dargs... = pb!!(dy, NoTangent(), df, dargs...)
        return df, dargs...
    end
    return y, intrinsic_function_pullback!!
end

macro inactive_intrinsic(name)
    return esc(:(
        function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{$name}}, args...)
            y = __intrinsic__(Val($name), map(primal, args)...)
            return CoDual(y, zero_tangent(y)), NoPullback()
        end
    ))
end

using .Intrinsics: abs_float
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{abs_float}}, x)
    abs_float_pullback!!(dy, df, dv, dx) = df, dv, dx + sign(primal(x)) * dy
    y = abs_float(primal(x))
    return CoDual(y, zero_tangent(y)), abs_float_pullback!!
end

using .Intrinsics: add_float
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{add_float}}, a, b)
    add_float_pb!!(c̄, f̄, v̄, ā, b̄) = f̄, v̄, c̄ + ā, c̄ + b̄
    c = add_float(primal(a), primal(b))
    return CoDual(c, zero_tangent(c)), add_float_pb!!
end

using .Intrinsics: add_float_fast
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{add_float_fast}}, a, b)
    add_float_fast_pb!!(c̄, f̄, v̄, ā, b̄) = f̄, v̄, add_float_fast(c̄, ā), add_float_fast(c̄, b̄)
    c = add_float_fast(primal(a), primal(b))
    return CoDual(c, zero_tangent(c)), add_float_fast_pb!!
end

@inactive_intrinsic Intrinsics.add_int

using .Intrinsics: add_ptr
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{add_ptr}}, a, b)
    throw(error("add_ptr intrinsic hit. This should never happen. Please open an issue"))
end

@inactive_intrinsic Intrinsics.and_int
@inactive_intrinsic Intrinsics.arraylen
@inactive_intrinsic Intrinsics.ashr_int

# atomic_fence
# atomic_pointermodify
# atomic_pointerref
# atomic_pointerreplace
# atomic_pointerset
# atomic_pointerswap

using .Intrinsics: bitcast
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{bitcast}}, T, x)
    _T = primal(T)
    _x = primal(x)
    v = bitcast(_T, _x)
    if _T <: Ptr && _x isa Ptr
        dv = bitcast(Ptr{tangent_type(eltype(_T))}, shadow(x))
    else
        dv = zero_tangent(v)
    end
    return CoDual(v, dv), NoPullback()
end

@inactive_intrinsic Intrinsics.bswap_int
@inactive_intrinsic Intrinsics.ceil_llvm
# cglobal
@inactive_intrinsic Intrinsics.checked_sadd_int
@inactive_intrinsic Intrinsics.checked_sdiv_int
@inactive_intrinsic Intrinsics.checked_smul_int
@inactive_intrinsic Intrinsics.checked_srem_int
@inactive_intrinsic Intrinsics.checked_ssub_int
@inactive_intrinsic Intrinsics.checked_uadd_int
@inactive_intrinsic Intrinsics.checked_udiv_int
@inactive_intrinsic Intrinsics.checked_umul_int
@inactive_intrinsic Intrinsics.checked_urem_int
@inactive_intrinsic Intrinsics.checked_usub_int

using .Intrinsics: copysign_float
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{copysign_float}}, x, y)
    _x = primal(x)
    _y = primal(y)
    function copysign_float_pullback!!(dz, df, dv, dx, dy)
        return df, dv, dx + dz * sign(_y), dy
    end
    z = copysign_float(_x, _y)
    return CoDual(z, zero_tangent(z)), copysign_float_pullback!!
end

@inactive_intrinsic Intrinsics.ctlz_int
@inactive_intrinsic Intrinsics.ctpop_int
@inactive_intrinsic Intrinsics.cttz_int

using .Intrinsics: div_float
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{div_float}}, a, b)
    _a = primal(a)
    _b = primal(b)
    _y = div_float(_a, _b)
    function div_float_pullback!!(dy, df, dv, da, db)
        da += div_float(dy, _b)
        db -= dy * _a / _b^2
        return df, dv, da, db
    end
    return CoDual(_y, zero_tangent(_y)), div_float_pullback!!
end

using .Intrinsics: div_float_fast
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{div_float_fast}}, a, b)
    _a = primal(a)
    _b = primal(b)
    _y = div_float_fast(_a, _b)
    function div_float_pullback!!(dy, df, dv, da, db)
        da += div_float_fast(dy, _b)
        db -= dy * div_float_fast(_a, _b^2)
        return df, dv, da, db
    end
    return CoDual(_y, zero_tangent(_y)), div_float_pullback!!
end

@inactive_intrinsic Intrinsics.eq_float
@inactive_intrinsic Intrinsics.eq_float_fast
@inactive_intrinsic Intrinsics.eq_int
@inactive_intrinsic Intrinsics.flipsign_int
@inactive_intrinsic Intrinsics.floor_llvm

# fma_float -- actually interesting
using .Intrinsics: fma_float
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{fma_float}}, x, y, z)
    _x = primal(x)
    _y = primal(y)
    _z = primal(z)
    function fma_float_pullback!!(da, df, dv, dx, dy, dz)
        return df, dv, fma_float(da, _y, dx), fma_float(da, _x, dy), dz + da
    end
    a = fma_float(_x, _y, _z)
    return CoDual(a, zero_tangent(a)), fma_float_pullback!!
end

# fpext -- maybe interesting

@inactive_intrinsic Intrinsics.fpiseq
@inactive_intrinsic Intrinsics.fptosi
@inactive_intrinsic Intrinsics.fptoui

# fptrunc -- maybe interesting

@inactive_intrinsic Intrinsics.have_fma
@inactive_intrinsic Intrinsics.le_float
@inactive_intrinsic Intrinsics.le_float_fast

# llvmcall -- interesting and not implementable at the minute

@inactive_intrinsic Intrinsics.lshr_int
@inactive_intrinsic Intrinsics.lt_float
@inactive_intrinsic Intrinsics.lt_float_fast

using .Intrinsics: mul_float
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{mul_float}}, a, b)
    _a = primal(a)
    _b = primal(b)
    function mul_float_pb!!(dc, df, dv, da, db)
        return df, dv, fma_float(dc, _b, da), fma_float(_a, dc, db)
    end
    c = mul_float(_a, _b)
    return CoDual(c, zero_tangent(c)), mul_float_pb!!
end

using .Intrinsics: mul_float_fast
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{mul_float_fast}}, a, b)
    _a = primal(a)
    _b = primal(b)
    function mul_float_pb!!(dc, df, dv, da, db)
        return df, dv, fma_float(dc, _b, da), fma_float(_a, dc, db)
    end
    c = mul_float_fast(_a, _b)
    return CoDual(c, zero_tangent(c)), mul_float_pb!!
end

@inactive_intrinsic Intrinsics.mul_int

using .Intrinsics: muladd_float
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{muladd_float}}, x, y, z)
    _x = primal(x)
    _y = primal(y)
    _z = primal(z)
    function muladd_float_pullback!!(da, df, dv, dx, dy, dz)
        return df, dv, dx + da * _y, dy + da * _x, dz + da
    end
    a = muladd_float(_x, _y, _z)
    return CoDual(a, zero_tangent(a)), muladd_float_pullback!!
end

@inactive_intrinsic Intrinsics.ne_float
@inactive_intrinsic Intrinsics.ne_float_fast
@inactive_intrinsic Intrinsics.ne_int

using .Intrinsics: neg_float
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{neg_float}}, x)
    _x = primal(x)
    neg_float_pullback!!(dy, df, dv, dx) = df, dv, sub_float(dx, dy)
    y = neg_float(_x)
    return CoDual(y, zero_tangent(y)), neg_float_pullback!!
end

using .Intrinsics: neg_float_fast
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{neg_float_fast}}, x)
    _x = primal(x)
    neg_float_fast_pullback!!(dy, df, dv, dx) = df, dv, sub_float_fast(dx, dy)
    y = neg_float_fast(_x)
    return CoDual(y, zero_tangent(y)), neg_float_fast_pullback!!
end

@inactive_intrinsic Intrinsics.neg_int
@inactive_intrinsic Intrinsics.not_int
@inactive_intrinsic Intrinsics.or_int

using .Intrinsics: pointerref
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{pointerref}}, x, y, z)
    _x = primal(x)
    _y = primal(y)
    _z = primal(z)
    x_s = shadow(x)
    a = CoDual(pointerref(_x, _y, _z), pointerref(x_s, _y, _z))
    function pointerref_pullback!!(da, df, dv, dx, dy, dz)
        dx_v = pointerref(dx, _y, _z)
        new_dx_v = increment!!(dx_v, da)
        pointerset(dx, new_dx_v, _y, _z)
        return df, dv, dx, dy, dz
    end
    return a, pointerref_pullback!!
end

using .Intrinsics: pointerset
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{pointerset}}, p, x, idx, z)
    _p = primal(p)
    _idx = primal(idx)
    _z = primal(z)
    old_value = pointerref(_p, _idx, _z)
    old_shadow = pointerref(shadow(p), _idx, _z)
    function pointerset_pullback!!(_, df, dv, dp, dx, didx, dz)
        dx_new = increment!!(dx, pointerref(dp, _idx, _z))
        pointerset(_p, old_value, _idx, _z)
        pointerset(dp, old_shadow, _idx, _z)
        return df, dv, dp, dx_new, didx, dz
    end
    pointerset(_p, primal(x), _idx, _z)
    pointerset(shadow(p), shadow(x), _idx, _z)
    return p, pointerset_pullback!!
end

# rem_float -- appears to be unused
# rem_float_fast -- appears to be unused
@inactive_intrinsic Intrinsics.rint_llvm
@inactive_intrinsic Intrinsics.sdiv_int
@inactive_intrinsic Intrinsics.sext_int
@inactive_intrinsic Intrinsics.shl_int
@inactive_intrinsic Intrinsics.sitofp
@inactive_intrinsic Intrinsics.sle_int
@inactive_intrinsic Intrinsics.slt_int

using .Intrinsics: sqrt_llvm
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{sqrt_llvm}}, x)
    _x = primal(x)
    llvm_sqrt_pullback!!(dy, df, dv, dx) = df, dv, dx + dy * inv(2 * sqrt(_x))
    return CoDual(sqrt_llvm(_x), zero(_x)), llvm_sqrt_pullback!!
end

using .Intrinsics: sqrt_llvm_fast
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{sqrt_llvm_fast}}, x)
    _x = primal(x)
    llvm_sqrt_pullback!!(dy, df, dv, dx) = df, dv, dx + dy * inv(2 * sqrt(_x))
    return CoDual(sqrt_llvm_fast(_x), zero(_x)), llvm_sqrt_pullback!!
end

@inactive_intrinsic Intrinsics.srem_int

using .Intrinsics: sub_float
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{sub_float}}, a, b)
    _a = primal(a)
    _b = primal(b)
    sub_float_pullback!!(dc, df, dv, da, db) = df, dv, add_float(da, dc), sub_float(db, dc)
    c = sub_float(_a, _b)
    return CoDual(c, zero_tangent(c)), sub_float_pullback!!
end

using .Intrinsics: sub_float_fast
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{sub_float_fast}}, a, b)
    _a = primal(a)
    _b = primal(b)
    function sub_float_fast_pullback!!(dc, df, dv, da, db)
        return df, dv, add_float_fast(da, dc), sub_float_fast(db, dc)
    end
    c = sub_float_fast(_a, _b)
    return CoDual(c, zero_tangent(c)), sub_float_fast_pullback!!
end

@inactive_intrinsic Intrinsics.sub_int

using .Intrinsics: sub_ptr
function rrule!!(::CoDual{typeof(__intrinsic__)}, ::CoDual{Val{sub_ptr}}, a, b)
    throw(error("sub_ptr intrinsic hit. This should never happen. Please open an issue"))
end

@inactive_intrinsic Intrinsics.trunc_int
@inactive_intrinsic Intrinsics.trunc_llvm
@inactive_intrinsic Intrinsics.udiv_int
@inactive_intrinsic Intrinsics.uitofp
@inactive_intrinsic Intrinsics.ule_int
@inactive_intrinsic Intrinsics.ult_int
@inactive_intrinsic Intrinsics.urem_int
@inactive_intrinsic Intrinsics.xor_int
@inactive_intrinsic Intrinsics.zext_int

function rrule!!(::CoDual{typeof(<:)}, T1, T2)
    return CoDual(<:(primal(T1), primal(T2)), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(===)}, args...)
    return CoDual(===(map(primal, args)...), NoTangent()), NoPullback()
end

# Core._abstracttype
# Core._apply_iterate
# Core._apply_pure
# Core._call_in_world
# Core._call_in_world_total
# Core._call_latest
# Core._compute_sparams
# Core._equiv_typedef
# Core._expr
# Core._primitivetype
# Core._setsuper!
# Core._structtype
# Core._svec_ref
# Core._typebody!
# Core._typevar

function rrule!!(::CoDual{typeof(Core._typevar)}, args...)
    y = Core._typevar(map(primal, args)...)
    return CoDual(y, zero_tangent(y)), NoPullback()
end

function rrule!!(::CoDual{typeof(Core.apply_type)}, args...)
    arg_primals = map(primal, args)
    T = Core.apply_type(arg_primals...)
    @show T
    return CoDual(T, zero_tangent(T)), NoPullback()
end

function rrule!!(
    ::CoDual{typeof(Core.arrayref)},
    inbounds::CoDual{Bool},
    x::CoDual{<:Array},
    inds::CoDual{Int}...,
)
    _inbounds = primal(inbounds)
    _inds = map(primal, inds)
    function arrayref_pullback!!(dy, df, dinbounds, dx, dinds...)
        current_val = arrayref(_inbounds, dx, _inds...)
        arrayset(_inbounds, dx, increment!!(current_val, dy), _inds...)
        return df, dinbounds, dx, dinds...
    end
    _y = arrayref(_inbounds, primal(x), _inds...)
    dy = arrayref(_inbounds, shadow(x), _inds...)
    return CoDual(_y, dy), arrayref_pullback!!
end

function rrule!!(
    ::CoDual{typeof(Core.arrayset)},
    inbounds::CoDual{Bool},
    A::CoDual{<:Array, TdA},
    v::CoDual,
    inds::CoDual{Int}...,
) where {V, TdA <: Array{V}}
    _inbounds = primal(inbounds)
    _inds = map(primal, inds)
    old_A_v = arrayref(_inbounds, primal(A), _inds...)
    old_A_v_t = arrayref(_inbounds, shadow(A), _inds...)
    arrayset(_inbounds, primal(A), primal(v), _inds...)
    arrayset(_inbounds, shadow(A), shadow(v), _inds...)
    function setindex_pullback!!(dA::TdA, df, dinbounds, dA2::TdA, dv, dinds::NoTangent...)
        dv_new = increment!!(dv, arrayref(_inbounds, dA, _inds...))
        arrayset(_inbounds, primal(A), old_A_v, _inds...)
        arrayset(_inbounds, dA, old_A_v_t, _inds...)
        return df, dinbounds, dA, dv_new, dinds...
    end
    return A, setindex_pullback!!
end

function rrule!!(::CoDual{typeof(Core.arraysize)}, X, dim)
    return CoDual(Core.arraysize(primal(X), primal(dim)), NoTangent()), NoPullback()
end

# Core.compilerbarrier
# Core.const_arrayref
# Core.donotdelete
# Core.finalizer
# Core.get_binding_type

function rrule!!(::CoDual{typeof(Core.ifelse)}, cond, a, b)
    _cond = primal(cond)
    function ifelse_pullback!!(dc, df, ::NoTangent, da, db)
        da = _cond ? increment!!(da, dc) : da
        db = _cond ? db : increment!!(db, dc)
        return df, NoTangent(), da, db
    end
    return ifelse(_cond, a, b), ifelse_pullback!!
end

# Core.set_binding_type!

function rrule!!(::CoDual{typeof(Core.sizeof)}, x)
    return CoDual(Core.sizeof(primal(x)), NoTangent()), NoPullback()
end

# Core.svec

function rrule!!(::CoDual{typeof(applicable)}, f, args...)
    return CoDual(applicable(primal(f), map(primal, args)...), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(Core.fieldtype)}, args...)
    arg_primals = map(primal, args)
    return CoDual(Core.fieldtype(arg_primals...), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(getfield)}, value::CoDual, name::CoDual)
    _name = primal(name)
    function getfield_pullback(dy, ::NoTangent, dvalue, ::NoTangent)
        new_dvalue = _increment_field!!(dvalue, dy, _name)
        return NoTangent(), new_dvalue, NoTangent()
    end
    y = CoDual(
        getfield(primal(value), _name),
        _get_shadow_field(primal(value), shadow(value), _name),
    )
    return y, getfield_pullback
end

function rrule!!(::CoDual{typeof(getfield)}, value::CoDual, name::CoDual, order::CoDual)
    _name = primal(name)
    _order = primal(order)
    function getfield_pullback(dy, df, dvalue, dname, dorder)
        new_dvalue = _increment_field!!(dvalue, dy, _name)
        return df, new_dvalue, dname, dorder
    end
    _order = _order isa Expr ? true : _order
    y = CoDual(
        getfield(primal(value), _name, _order),
        _get_shadow_field(primal(value), shadow(value), _name, _order),
    )
    return y, getfield_pullback
end

_get_shadow_field(_, shadow, f...) = getfield(shadow, f...)
function _get_shadow_field(_, shadow::Union{Tangent, MutableTangent}, f...)
    return _value(getfield(shadow.fields, f...))
end
_get_shadow_field(primal, shadow::NoTangent, f...) = uninit_tangent(getfield(primal, f...))

_increment_field!!(x, y, f) = increment_field!!(x, y, f)
_increment_field!!(x::NoTangent, y, f) = x

uninit_tangent(x) = zero_tangent(x)
uninit_tangent(x::Ptr{P}) where {P} = bitcast(Ptr{tangent_type(P)}, x)

function rrule!!(::CoDual{typeof(getglobal)}, a, b)
    v = getglobal(primal(a), primal(b))
    return CoDual(v, zero_tangent(v)), NoPullback()
end

# invoke

function rrule!!(::CoDual{typeof(isa)}, x, T)
    return CoDual(isa(primal(x), primal(T)), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(isdefined)}, args...)
    return CoDual(isdefined(map(primal, args)...), NoTangent()), NoPullback()
end

# modifyfield!

function rrule!!(::CoDual{typeof(nfields)}, x)
    return CoDual(nfields(primal(x)), NoTangent()), NoPullback()
end

# replacefield!

function _setfield!(value::MutableTangent, name, x)
    @set value.fields.$name = x
    return x
end

function rrule!!(::CoDual{typeof(setfield!)}, value, name, x)
    _name = primal(name)
    old_x = isdefined(primal(value), _name) ? getfield(primal(value), _name) : nothing
    function setfield!_pullback(dy, df, dvalue, ::NoTangent, dx)
        new_dx = increment!!(dx, getfield(dvalue.fields, _name).tangent)
        set_field_to_zero!!(dvalue, _name)
        new_dx = increment!!(new_dx, dy)
        old_x !== nothing && setfield!(primal(value), _name, old_x)
        return df, dvalue, NoTangent(), new_dx
    end
    y = CoDual(
        setfield!(primal(value), _name, primal(x)),
        _setfield!(shadow(value), _name, shadow(x)),
    )
    return y, setfield!_pullback
end

# swapfield!
# throw

function rrule!!(::CoDual{typeof(tuple)}, args...)
    y = CoDual(tuple(map(primal, args)...), tuple(map(shadow, args)...))
    tuple_pullback(dy, ::NoTangent, dargs...) = NoTangent(), map(increment!!, dargs, dy)...
    return y, tuple_pullback
end

function rrule!!(::CoDual{typeof(typeassert)}, x, type)
    function typeassert_pullback(dy, ::NoTangent, dx, ::NoTangent)
        return NoTangent(), increment!!(dx, dy), NoTangent()
    end
    return CoDual(typeassert(primal(x), primal(type)), shadow(x)), typeassert_pullback
end

rrule!!(::CoDual{typeof(typeof)}, x) = CoDual(typeof(primal(x)), NoTangent()), NoPullback()
