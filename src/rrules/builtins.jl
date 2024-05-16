
#
# Core.Builtin -- these are "primitive" functions which must have rrules because no IR
# is available.
#
# There is a finite number of these functions.
# Any built-ins which don't have rules defined are left as comments with their names
# in this block of code
# As of version 1.9.2 of Julia, there are exactly 139 examples of `Core.Builtin`s.
#


@is_primitive MinimalCtx Tuple{Core.Builtin, Vararg}

module IntrinsicsWrappers

using Base: IEEEFloat
using Core: Intrinsics
using Tapir
import ..Tapir:
    rrule!!, CoDual, primal, tangent, zero_tangent, NoPullback,
    tangent_type, increment!!, @is_primitive, MinimalCtx, is_primitive, NoFData,
    zero_rdata, NoRData, tuple_map, fdata, NoRData, rdata, increment_rdata!!, zero_fcodual

# Note: performance is not considered _at_ _all_ in this implementation.
function rrule!!(f::CoDual{<:Core.IntrinsicFunction}, args...)
    return rrule!!(CoDual(translate(Val(primal(f))), tangent(f)), args...)
end

macro intrinsic(name)
    expr = quote
        $name(x...) = Intrinsics.$name(x...)
        (is_primitive)(::Type{MinimalCtx}, ::Type{<:Tuple{typeof($name), Vararg}}) = true
        translate(::Val{Intrinsics.$name}) = $name
    end
    return esc(expr)
end

macro inactive_intrinsic(name)
    expr = quote
        $name(x...) = Intrinsics.$name(x...)
        (is_primitive)(::Type{MinimalCtx}, ::Type{<:Tuple{typeof($name), Vararg}}) = true
        translate(::Val{Intrinsics.$name}) = $name
        function rrule!!(f::CoDual{typeof($name)}, args::Vararg{Any, N}) where {N}
            return zero_fcodual($name(map(primal, args)...)), NoPullback(f, args...)
        end
    end
    return esc(expr)
end

@intrinsic abs_float
function rrule!!(::CoDual{typeof(abs_float)}, x)
    abs_float_pullback!!(dy) = NoRData(), sign(primal(x)) * dy
    y = abs_float(primal(x))
    return CoDual(y, NoFData()), abs_float_pullback!!
end

@intrinsic add_float
function rrule!!(::CoDual{typeof(add_float)}, a, b)
    add_float_pb!!(c̄) = NoRData(), c̄, c̄
    c = add_float(primal(a), primal(b))
    return CoDual(c, NoFData()), add_float_pb!!
end

@intrinsic add_float_fast
function rrule!!(::CoDual{typeof(add_float_fast)}, a, b)
    add_float_fast_pb!!(c̄) = NoRData(), c̄, c̄
    c = add_float_fast(primal(a), primal(b))
    return CoDual(c, NoFData()), add_float_fast_pb!!
end

@inactive_intrinsic add_int

@intrinsic add_ptr
function rrule!!(::CoDual{typeof(add_ptr)}, a, b)
    throw(error("add_ptr intrinsic hit. This should never happen. Please open an issue"))
end

@inactive_intrinsic and_int
@inactive_intrinsic arraylen
@inactive_intrinsic ashr_int

# atomic_fence
# atomic_pointermodify
# atomic_pointerref
# atomic_pointerreplace
# atomic_pointerset
# atomic_pointerswap

@intrinsic bitcast
function rrule!!(f::CoDual{typeof(bitcast)}, t::CoDual{Type{T}}, x::CoDual{V}) where {T, V}
    if T <: IEEEFloat
        msg = "It is not permissible to bitcast to a differentiable type during " *
        "AD, as this risks dropping tangents, and therefore risks silently giving the " *
        " wrong answer. If this call to bitcast appears as part of the implementation of " *
        "a differentiable function, you should write a rule for this function, or modify " *
        "its implementation to avoid the bitcast."
        throw(ArgumentError(msg))
    end
    _x = primal(x)
    v = bitcast(T, _x)
    if T <: Ptr && _x isa Ptr
        dv = bitcast(Ptr{tangent_type(eltype(T))}, tangent(x))
    else
        dv = NoFData()
    end
    return CoDual(v, dv), NoPullback(f, t, x)
end

@inactive_intrinsic bswap_int
@inactive_intrinsic ceil_llvm

#=
Replacement for `Core.Intrinsics.cglobal`. `cglobal` is different from the other intrinsics
in that the name `cglobal` is reversed by the language (try creating a variable called
`cglobal` -- Julia will not let you). Additionally, it requires that its first argument,
the specification of the name of the C cglobal variable that this intrinsic returns a
pointer to, is known statically. In this regard it is like foreigncalls.

As a consequence, it requires special handling. The name is converted into a `Val` so that
it is available statically, and the function into which `cglobal` calls are converted is
named `Tapir.IntrinsicsWrappers.__cglobal`, rather than `Tapir.IntrinsicsWrappers.cglobal`.

If you examine the code associated with `Tapir.intrinsic_to_function`, you will see that
special handling of `cglobal` is used.
=#
__cglobal(::Val{s}, x::Vararg{Any, N}) where {s, N} = cglobal(s, x...)

translate(::Val{Intrinsics.cglobal}) = __cglobal
Tapir.is_primitive(::Type{MinimalCtx}, ::Type{<:Tuple{typeof(__cglobal), Vararg}}) = true
function rrule!!(f::CoDual{typeof(__cglobal)}, args...)
    return Tapir.uninit_fcodual(__cglobal(map(primal, args)...)), NoPullback(f, args...)
end

@inactive_intrinsic checked_sadd_int
@inactive_intrinsic checked_sdiv_int
@inactive_intrinsic checked_smul_int
@inactive_intrinsic checked_srem_int
@inactive_intrinsic checked_ssub_int
@inactive_intrinsic checked_uadd_int
@inactive_intrinsic checked_udiv_int
@inactive_intrinsic checked_umul_int
@inactive_intrinsic checked_urem_int
@inactive_intrinsic checked_usub_int

@intrinsic copysign_float
function rrule!!(::CoDual{typeof(copysign_float)}, x, y)
    _x = primal(x)
    _y = primal(y)
    copysign_float_pullback!!(dz) = NoRData(), dz * sign(_y), zero_rdata(_y)
    z = copysign_float(_x, _y)
    return CoDual(z, NoFData()), copysign_float_pullback!!
end

@inactive_intrinsic ctlz_int
@inactive_intrinsic ctpop_int
@inactive_intrinsic cttz_int

@intrinsic div_float
function rrule!!(::CoDual{typeof(div_float)}, a, b)
    _a = primal(a)
    _b = primal(b)
    _y = div_float(_a, _b)
    div_float_pullback!!(dy) = NoRData(), div_float(dy, _b), -dy * _a / _b^2
    return CoDual(_y, NoFData()), div_float_pullback!!
end

@intrinsic div_float_fast
function rrule!!(::CoDual{typeof(div_float_fast)}, a, b)
    _a = primal(a)
    _b = primal(b)
    _y = div_float_fast(_a, _b)
    function div_float_pullback!!(dy)
        return NoRData(), div_float_fast(dy, _b), -dy * div_float_fast(_a, _b^2)
    end
    return CoDual(_y, NoFData()), div_float_pullback!!
end

@inactive_intrinsic eq_float
@inactive_intrinsic eq_float_fast
@inactive_intrinsic eq_int
@inactive_intrinsic flipsign_int
@inactive_intrinsic floor_llvm

@intrinsic fma_float
function rrule!!(::CoDual{typeof(fma_float)}, x, y, z)
    _x = primal(x)
    _y = primal(y)
    fma_float_pullback!!(da) = NoRData(), da * _y, da * _x, da
    return CoDual(fma_float(_x, _y, primal(z)), NoFData()), fma_float_pullback!!
end

# fpext -- maybe interesting

@inactive_intrinsic fpiseq

@intrinsic fptosi
function rrule!!(::CoDual{typeof(fptosi)}, ::CoDual...)
    msg = "It is not permissible to cast a float to a signed integer in " *
        "AD, as this risks dropping tangents, and therefore risks silently giving the " *
        " wrong answer. If this call to Core.Intrinsics.fptosi appears as part of the " *
        "implementation of a differentiable function, you should write a rule for this " *
        "function, or modify its implementation to avoid this call."
    throw(ArgumentError(msg))
end

@intrinsic fptoui
function rrule!!(::CoDual{typeof(fptoui)}, ::CoDual...)
    msg = "It is not permissible to cast a float to an unsigned integer in " *
        "AD, as this risks dropping tangents, and therefore risks silently giving the " *
        " wrong answer. If this call to Core.Intrinsics.fptoui appears as part of the " *
        "implementation of a differentiable function, you should write a rule for this " *
        "function, or modify its implementation to avoid this call."
    throw(ArgumentError(msg))
end

# fptrunc -- maybe interesting

@inactive_intrinsic have_fma
@inactive_intrinsic le_float
@inactive_intrinsic le_float_fast

# llvmcall -- interesting and not implementable at the minute

@inactive_intrinsic lshr_int
@inactive_intrinsic lt_float
@inactive_intrinsic lt_float_fast

@intrinsic mul_float
function rrule!!(::CoDual{typeof(mul_float)}, a, b)
    _a = primal(a)
    _b = primal(b)
    mul_float_pb!!(dc) = NoRData(), dc * _b, _a * dc
    return CoDual(mul_float(_a, _b), NoFData()), mul_float_pb!!
end

@intrinsic mul_float_fast
function rrule!!(::CoDual{typeof(mul_float_fast)}, a, b)
    _a = primal(a)
    _b = primal(b)
    mul_float_fast_pb!!(dc) = NoRData(), dc * _b, _a * dc
    return CoDual(mul_float_fast(_a, _b), NoFData()), mul_float_fast_pb!!
end

@inactive_intrinsic mul_int

@intrinsic muladd_float
function rrule!!(::CoDual{typeof(muladd_float)}, x, y, z)
    _x = primal(x)
    _y = primal(y)
    _z = primal(z)
    muladd_float_pullback!!(da) = NoRData(), da * _y, da * _x, da
    return CoDual(muladd_float(_x, _y, _z), NoFData()), muladd_float_pullback!!
end

@inactive_intrinsic ne_float
@inactive_intrinsic ne_float_fast
@inactive_intrinsic ne_int

@intrinsic neg_float
function rrule!!(::CoDual{typeof(neg_float)}, x)
    _x = primal(x)
    neg_float_pullback!!(dy) = NoRData(), -dy
    return CoDual(neg_float(_x), NoFData()), neg_float_pullback!!
end

@intrinsic neg_float_fast
function rrule!!(::CoDual{typeof(neg_float_fast)}, x)
    _x = primal(x)
    neg_float_fast_pullback!!(dy) = NoRData(), -dy
    return CoDual(neg_float_fast(_x), NoFData()), neg_float_fast_pullback!!
end

@inactive_intrinsic neg_int
@inactive_intrinsic not_int
@inactive_intrinsic or_int

@intrinsic pointerref
function rrule!!(::CoDual{typeof(pointerref)}, x, y, z)
    _x = primal(x)
    _y = primal(y)
    _z = primal(z)
    dx = tangent(x)
    a = CoDual(pointerref(_x, _y, _z), fdata(pointerref(dx, _y, _z)))
    if Tapir.rdata_type(tangent_type(Tapir._typeof(primal(a)))) == NoRData
        return a, NoPullback((NoRData(), NoRData(), NoRData(), NoRData()))
    else
        function pointerref_pullback!!(da)
            pointerset(dx, increment_rdata!!(pointerref(dx, _y, _z), da), _y, _z)
            return NoRData(), NoRData(), NoRData(), NoRData()
        end
        return a, pointerref_pullback!!
    end
end

@intrinsic pointerset
function rrule!!(::CoDual{typeof(pointerset)}, p, x, idx, z)
    _p = primal(p)
    _idx = primal(idx)
    _z = primal(z)
    old_value = pointerref(_p, _idx, _z)
    old_tangent = pointerref(tangent(p), _idx, _z)
    dp = tangent(p)
    function pointerset_pullback!!(::NoRData)
        dx_r = pointerref(dp, _idx, _z)
        pointerset(_p, old_value, _idx, _z)
        pointerset(dp, old_tangent, _idx, _z)
        return NoRData(), NoRData(), rdata(dx_r), NoRData(), NoRData()
    end
    pointerset(_p, primal(x), _idx, _z)
    pointerset(dp, zero_tangent(primal(x)), _idx, _z)
    return p, pointerset_pullback!!
end

# rem_float -- appears to be unused
# rem_float_fast -- appears to be unused
@inactive_intrinsic rint_llvm
@inactive_intrinsic sdiv_int
@inactive_intrinsic sext_int
@inactive_intrinsic shl_int
@inactive_intrinsic sitofp
@inactive_intrinsic sle_int
@inactive_intrinsic slt_int

@intrinsic sqrt_llvm
function rrule!!(::CoDual{typeof(sqrt_llvm)}, x)
    _x = primal(x)
    llvm_sqrt_pullback!!(dy) = NoRData(), dy * inv(2 * sqrt(_x))
    return CoDual(sqrt_llvm(_x), NoFData()), llvm_sqrt_pullback!!
end

@intrinsic sqrt_llvm_fast
function rrule!!(::CoDual{typeof(sqrt_llvm_fast)}, x)
    _x = primal(x)
    llvm_sqrt_fast_pullback!!(dy) = NoRData(), dy * inv(2 * sqrt(_x))
    return CoDual(sqrt_llvm_fast(_x), NoFData()), llvm_sqrt_fast_pullback!!
end

@inactive_intrinsic srem_int

@intrinsic sub_float
function rrule!!(::CoDual{typeof(sub_float)}, a, b)
    _a = primal(a)
    _b = primal(b)
    sub_float_pullback!!(dc) = NoRData(), dc, -dc
    return CoDual(sub_float(_a, _b), NoFData()), sub_float_pullback!!
end

@intrinsic sub_float_fast
function rrule!!(::CoDual{typeof(sub_float_fast)}, a, b)
    _a = primal(a)
    _b = primal(b)
    sub_float_fast_pullback!!(dc) = NoRData(), dc, -dc
    return CoDual(sub_float_fast(_a, _b), NoFData()), sub_float_fast_pullback!!
end

@inactive_intrinsic sub_int

@intrinsic sub_ptr
function rrule!!(::CoDual{typeof(sub_ptr)}, a, b)
    throw(error("sub_ptr intrinsic hit. This should never happen. Please open an issue"))
end

@inactive_intrinsic trunc_int
@inactive_intrinsic trunc_llvm
@inactive_intrinsic udiv_int
@inactive_intrinsic uitofp
@inactive_intrinsic ule_int
@inactive_intrinsic ult_int
@inactive_intrinsic urem_int
@inactive_intrinsic xor_int
@inactive_intrinsic zext_int

end # IntrinsicsWrappers

function rrule!!(f::CoDual{typeof(<:)}, T1, T2)
    return zero_fcodual(<:(primal(T1), primal(T2))), NoPullback(f, T1, T2)
end

function rrule!!(f::CoDual{typeof(===)}, x, y)
    return zero_fcodual(primal(x) === primal(y)), NoPullback(f, x, y)
end

# Core._abstracttype

#
# Core._apply_iterate
#
# Core._apply_iterate is a tricky case, and requires calling back into AD to handle
# properly. The basic strategy is to differentiate a function which is semantically
# identical to Core._apply_iterate, but whose components we know how to differentiate.
#

# A function with the same semantics as `Core._apply_iterate`, but which is differentiable.
function _apply_iterate_equivalent(itr, f::F, args::Vararg{Any, N}) where {F, N}
    vec_args = reduce(vcat, tuple_map(collect, args))
    tuple_args = __vec_to_tuple(vec_args)
    return __barrier(f, tuple_args)
end

# A primitive used to avoid exposing `_apply_iterate_equivalent` to `Core._apply_iterate`.
__vec_to_tuple(v::Vector) = Tuple(v)

@is_primitive MinimalCtx Tuple{typeof(__vec_to_tuple), Vector}

function rrule!!(::CoDual{typeof(__vec_to_tuple)}, v::CoDual{<:Vector})
    dv = tangent(v)
    y = CoDual(Tuple(primal(v)), fdata(Tuple(dv)))
    function vec_to_tuple_pb!!(dy::Union{Tuple, NoRData})
        if dy isa Tuple
            for n in eachindex(dy)
                dv[n] = increment_rdata!!(dv[n], dy[n])
            end
        end
        return NoRData(), NoRData()
    end
    return y, vec_to_tuple_pb!!
end

@noinline __barrier(f::F, args::Tuple) where {F} = f(args...)

# Over-ride default definition of `is_primitive` for buildins.
is_primitive(::Type{MinimalCtx}, ::Type{<:Tuple{typeof(Core._apply_iterate), Vararg}}) = false

struct ApplyIterateRule{R}
    rule::R
end

function (rule::ApplyIterateRule)(::CoDual{typeof(Core._apply_iterate)}, args::CoDual...)
    return rule.rule(zero_fcodual(_apply_iterate_equivalent), args...)
end

function build_rrule(
    interp::TapirInterpreter, sig::Type{<:Tuple{typeof(Core._apply_iterate), Vararg}};
    kwargs...
)
    new_sig = Tuple{typeof(_apply_iterate_equivalent), sig.parameters[2:end]...}
    return ApplyIterateRule(build_rrule(interp, new_sig; kwargs...))
end

function rule_type(
    interp::TapirInterpreter{C}, sig::Type{<:Tuple{typeof(Core._apply_iterate), Vararg}}
) where {C}
    new_sig = Tuple{typeof(_apply_iterate_equivalent), sig.parameters[2:end]...}
    return ApplyIterateRule{rule_type(interp, new_sig)}
end

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

function rrule!!(f::CoDual{typeof(Core._typevar)}, args...)
    return zero_fcodual(Core._typevar(map(primal, args)...)), NoPullback(f, args...)
end

function rrule!!(f::CoDual{typeof(Core.apply_type)}, args...)
    T = Core.apply_type(tuple_map(primal, args)...)
    return CoDual{_typeof(T), NoFData}(T, NoFData()), NoPullback(f, args...)
end

Base.@propagate_inbounds function rrule!!(
    ::CoDual{typeof(Core.arrayref)},
    checkbounds::CoDual{Bool},
    x::CoDual{<:Array},
    inds::Vararg{CoDual{Int}, N},
) where {N}

    # Convert to linear indices to reduce amount of data required on the reverse-pass, to
    # avoid converting from cartesian to linear indices multiple times, and to perform a
    # bounds check if required by the calling context.
    lin_inds = LinearIndices(size(primal(x)))[tuple_map(primal, inds)...]

    dx = tangent(x)
    function arrayref_pullback!!(dy)
        new_tangent = increment_rdata!!(arrayref(false, dx, lin_inds), dy)
        arrayset(false, dx, new_tangent, lin_inds)
        return NoRData(), NoRData(), NoRData(), ntuple(_ -> NoRData(), N)...
    end
    _y = arrayref(false, primal(x), lin_inds)
    dy = fdata(arrayref(false, tangent(x), lin_inds))
    return CoDual(_y, dy), arrayref_pullback!!
end

function rrule!!(
    ::CoDual{typeof(Core.arrayset)},
    inbounds::CoDual{Bool},
    A::CoDual{<:Array{P}, TdA},
    v::CoDual,
    inds::CoDual{Int}...,
) where {P, V, TdA <: Array{V}}
    _inbounds = primal(inbounds)
    _inds = map(primal, inds)

    if isbitstype(P)
        return isbits_arrayset_rrule(_inbounds, _inds, A, v)
    end

    to_save = isassigned(primal(A), _inds...)
    old_A = Ref{Tuple{P, V}}()
    if to_save
        old_A[] = (
            arrayref(_inbounds, primal(A), _inds...),
            arrayref(_inbounds, tangent(A), _inds...),
        )
    end

    arrayset(_inbounds, primal(A), primal(v), _inds...)
    dA = tangent(A)
    arrayset(_inbounds, dA, tangent(tangent(v), zero_rdata(primal(v))), _inds...)
    function arrayset_pullback!!(::NoRData)
        dv = rdata(arrayref(_inbounds, dA, _inds...))
        if to_save
            arrayset(_inbounds, primal(A), old_A[][1], _inds...)
            arrayset(_inbounds, dA, old_A[][2], _inds...)
        end
        return NoRData(), NoRData(), NoRData(), dv, tuple_map(_ -> NoRData(), _inds)...
    end
    return A, arrayset_pullback!!
end

function isbits_arrayset_rrule(
    boundscheck, _inds, A::CoDual{<:Array{P}, TdA}, v::CoDual{P}
) where {P, V, TdA <: Array{V}}

    # Convert to linear indices
    lin_inds = LinearIndices(size(primal(A)))[_inds...]

    old_A = (arrayref(false, primal(A), lin_inds), arrayref(false, tangent(A), lin_inds))
    arrayset(false, primal(A), primal(v), lin_inds)

    _A = primal(A)
    dA = tangent(A)
    arrayset(false, dA, zero_tangent(primal(v)), lin_inds)
    ninds = Val(length(_inds))
    function isbits_arrayset_pullback!!(::NoRData)
        dv = rdata(arrayref(false, dA, lin_inds))
        arrayset(false, _A, old_A[1], lin_inds)
        arrayset(false, dA, old_A[2], lin_inds)
        return NoRData(), NoRData(), NoRData(), dv, tuple_fill(NoRData(), ninds)...
    end
    return A, isbits_arrayset_pullback!!
end

function rrule!!(f::CoDual{typeof(Core.arraysize)}, X, dim)
    return zero_fcodual(Core.arraysize(primal(X), primal(dim))), NoPullback(f, X, dim)
end

# Core.compilerbarrier
# Core.const_arrayref
# Core.donotdelete
# Core.finalizer
# Core.get_binding_type

function rrule!!(f::CoDual{typeof(Core.ifelse)}, cond, a::A, b::B) where {A, B}
    _cond = primal(cond)
    p_a = primal(a)
    p_b = primal(b)
    pb!! = if rdata_type(tangent_type(A)) == NoRData && rdata_type(tangent_type(B)) == NoRData
        NoPullback(f, cond, a, b)
    else
        lazy_da = LazyZeroRData(p_a)
        lazy_db = LazyZeroRData(p_b)
        function ifelse_pullback!!(dc)
            da = ifelse(_cond, dc, instantiate(lazy_da))
            db = ifelse(_cond, instantiate(lazy_db), dc)
            return NoRData(), NoRData(), da, db
        end
    end

    # It's a good idea to split up applying ifelse to the primal and tangent. This is
    # because if you push a `CoDual` through ifelse, it _forces_ the construction of the
    # CoDual. Conversely, if you pass through the primal and tangents separately, the
    # compiler will often be able to avoid constructing the CoDual at all by inlining lots
    # of stuff away.
    return CoDual(ifelse(_cond, p_a, p_b), ifelse(_cond, tangent(a), tangent(b))), pb!!
end

# Core.set_binding_type!

function rrule!!(f::CoDual{typeof(Core.sizeof)}, x)
    return zero_fcodual(Core.sizeof(primal(x))), NoPullback(f, x)
end

# Core.svec

function rrule!!(_f::CoDual{typeof(applicable)}, f, args...)
    pb!! = NoPullback(_f, f, args...)
    return zero_fcodual(applicable(primal(f), map(primal, args)...)), pb!!
end

function rrule!!(f::CoDual{typeof(Core.fieldtype)}, args::Vararg{Any, N}) where {N}
    arg_primals = tuple_map(primal, args)
    return CoDual(Core.fieldtype(arg_primals...), NoFData()), NoPullback(f, args...)
end

function rrule!!(f::CoDual{typeof(getfield)}, x::CoDual{P}, name::CoDual) where {P}
    if tangent_type(P) == NoTangent
        y = uninit_fcodual(getfield(primal(x), primal(name)))
        return y, NoPullback(f, x, name)
    elseif is_homogeneous_and_immutable(primal(x))
        dx_r = LazyZeroRData(primal(x))
        _name = primal(name)
        function immutable_lgetfield_pb!!(dy)
            return NoRData(), increment_field!!(instantiate(dx_r), dy, _name), NoRData()
        end
        yp = getfield(primal(x), _name)
        y = CoDual(yp, _get_fdata_field(primal(x), tangent(x), _name))
        return y, immutable_lgetfield_pb!!
    else
        return rrule!!(uninit_fcodual(lgetfield), x, uninit_fcodual(Val(primal(name))))
    end
end

function rrule!!(f::CoDual{typeof(getfield)}, x::CoDual{P}, name::CoDual, order::CoDual) where {P}
    if tangent_type(P) == NoTangent
        y = uninit_fcodual(getfield(primal(x), primal(name)))
        return y, NoPullback(f, x, name, order)
    elseif is_homogeneous_and_immutable(primal(x))
        dx_r = LazyZeroRData(primal(x))
        _name = primal(name)
        function immutable_lgetfield_pb!!(dy)
            tmp = increment_field!!(instantiate(dx_r), dy, _name)
            return NoRData(), tmp, NoRData(), NoRData()
        end
        yp = getfield(primal(x), _name, primal(order))
        y = CoDual(yp, _get_fdata_field(primal(x), tangent(x), _name))
        return y, immutable_lgetfield_pb!!
    else
        literal_name = uninit_fcodual(Val(primal(name)))
        literal_order = uninit_fcodual(Val(primal(order)))
        return rrule!!(uninit_fcodual(lgetfield), x, literal_name, literal_order)
    end
end

@generated is_homogeneous_and_immutable(::P) where {P<:Tuple} = allequal(P.parameters)
@inline is_homogeneous_and_immutable(p::NamedTuple) = is_homogeneous_and_immutable(Tuple(p))
is_homogeneous_and_immutable(::Any) = false

# # Highly specialised rrule to handle tuples of DataTypes.
# function rrule!!(::CoDual{typeof(getfield)}, value::CoDual{P}, name::CoDual) where {P<:NTuple{<:Any, DataType}}
#     pb!! = NoPullback((NoRData(), NoRData(), NoRData(), NoRData()))
#     y = CoDual{DataType, NoFData}(getfield(primal(value), primal(name)), NoFData())
#     return y, pb!!
# end
# function rrule!!(::CoDual{typeof(getfield)}, value::CoDual{P}, name::CoDual, order::CoDual) where {P<:NTuple{<:Any, DataType}}
#     pb!! = NoPullback((NoRData(), NoRData(), NoRData(), NoRData()))
#     y = CoDual{DataType, NoFData}(getfield(primal(value), primal(name), primal(order)), NoFData())
#     return y, pb!!
# end

function rrule!!(f::CoDual{typeof(getglobal)}, a, b)
    return zero_fcodual(getglobal(primal(a), primal(b))), NoPullback(f, a, b)
end

# invoke

function rrule!!(f::CoDual{typeof(isa)}, x, T)
    return zero_fcodual(isa(primal(x), primal(T))), NoPullback(f, x, T)
end

function rrule!!(f::CoDual{typeof(isdefined)}, args...)
    return zero_fcodual(isdefined(map(primal, args)...)), NoPullback(f, args...)
end

# modifyfield!

rrule!!(f::CoDual{typeof(nfields)}, x) = zero_fcodual(nfields(primal(x))), NoPullback(f, x)

# replacefield!

function rrule!!(::CoDual{typeof(setfield!)}, value, name, x)
    literal_name = uninit_fcodual(Val(primal(name)))
    return rrule!!(uninit_fcodual(lsetfield!), value, literal_name, x)
end

# swapfield!
# throw

struct TuplePullback{N} end

@inline (::TuplePullback{N})(dy::Tuple) where {N} = NoRData(), dy...

@inline function (::TuplePullback{N})(::NoRData) where {N}
    return NoRData(), ntuple(_ -> NoRData(), N)...
end

@inline tuple_pullback(dy) = NoRData(), dy...

@inline tuple_pullback(dy::NoRData) = NoRData()

function rrule!!(f::CoDual{typeof(tuple)}, args::Vararg{Any, N}) where {N}
    primal_output = tuple(map(primal, args)...)
    if tangent_type(_typeof(primal_output)) == NoTangent
        return zero_fcodual(primal_output), NoPullback(f, args...)
    else
        if fdata_type(tangent_type(_typeof(primal_output))) == NoFData
            return zero_fcodual(primal_output), TuplePullback{N}()
        else
            return CoDual(primal_output, tuple(map(tangent, args)...)), TuplePullback{N}()
        end
    end
end

function rrule!!(::CoDual{typeof(typeassert)}, x::CoDual, type::CoDual)
    typeassert_pullback(dy) = NoRData(), dy, NoRData()
    return CoDual(typeassert(primal(x), primal(type)), tangent(x)), typeassert_pullback
end

function rrule!!(f::CoDual{typeof(typeof)}, x::CoDual)
    return zero_fcodual(typeof(primal(x))), NoPullback(f, x)
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:builtins})

    _x = Ref(5.0) # data used in tests which aren't protected by GC.
    _dx = Ref(4.0)
    _a = Vector{Vector{Float64}}(undef, 3)
    _a[1] = [5.4, 4.23, -0.1, 2.1]

    x = randn(5)
    p = pointer(x)
    dx = randn(5)
    dp = pointer(dx)

    y = [1, 2, 3]
    q = pointer(y)
    dy = zero_tangent(y)
    dq = pointer(dy)

    # Slightly wider range for builtins whose performance is known not to be great.
    _range = (lb=1e-3, ub=200.0)

    test_cases = Any[

        # Core.Intrinsics:
        (false, :stability, nothing, IntrinsicsWrappers.abs_float, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.add_float, 4.0, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.add_float_fast, 4.0, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.add_int, 1, 2),
        (false, :stability, nothing, IntrinsicsWrappers.and_int, 2, 3),
        (false, :stability, nothing, IntrinsicsWrappers.arraylen, randn(10)),
        (false, :stability, nothing, IntrinsicsWrappers.arraylen, randn(10, 7)),
        (false, :stability, nothing, IntrinsicsWrappers.ashr_int, 123456, 0x0000000000000020),
        # atomic_fence -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointermodify -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerref -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerreplace -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerset -- NEEDS IMPLEMENTING AND TESTING
        # atomic_pointerswap -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, IntrinsicsWrappers.bitcast, UInt64, 5),
        (false, :stability, nothing, IntrinsicsWrappers.bswap_int, 5),
        (false, :stability, nothing, IntrinsicsWrappers.ceil_llvm, 4.1),
        (
            true,
            :stability,
            nothing,
            IntrinsicsWrappers.__cglobal,
            Val{:jl_uv_stdout}(),
            Ptr{Cvoid},
        ),
        (false, :stability, nothing, IntrinsicsWrappers.checked_sadd_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_sdiv_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_smul_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_srem_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_ssub_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_uadd_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_udiv_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_umul_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_urem_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.checked_usub_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.copysign_float, 5.0, -3.0),
        [false, :stability, nothing, IntrinsicsWrappers.ctlz_int, 5],
        (false, :stability, nothing, IntrinsicsWrappers.ctpop_int, 5),
        (false, :stability, nothing, IntrinsicsWrappers.cttz_int, 5),
        (false, :stability, nothing, IntrinsicsWrappers.div_float, 5.0, 3.0),
        (false, :stability, nothing, IntrinsicsWrappers.div_float_fast, 5.0, 3.0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float, 4.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float_fast, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_float_fast, 4.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.eq_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.eq_int, 4, 4),
        (false, :stability, nothing, IntrinsicsWrappers.flipsign_int, 4, -3),
        (false, :stability, nothing, IntrinsicsWrappers.floor_llvm, 4.1),
        (false, :stability, nothing, IntrinsicsWrappers.fma_float, 5.0, 4.0, 3.0),
        # fpext -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, IntrinsicsWrappers.fpiseq, 4.1, 4.0),
        # fptrunc -- maybe interesting
        (true, :stability, nothing, IntrinsicsWrappers.have_fma, Float64),
        (false, :stability, nothing, IntrinsicsWrappers.le_float, 4.1, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.le_float_fast, 4.1, 4.0),
        # llvm_call -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, IntrinsicsWrappers.lshr_int, 1308622848, 0x0000000000000018),
        (false, :stability, nothing, IntrinsicsWrappers.lt_float, 4.1, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.lt_float_fast, 4.1, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.mul_float, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.mul_float_fast, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.mul_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.muladd_float, 5.0, 4.0, 3.0),
        (false, :stability, nothing, IntrinsicsWrappers.ne_float, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.ne_float_fast, 5.0, 4.0),
        (false, :stability, nothing, IntrinsicsWrappers.ne_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.ne_int, 5, 5),
        (false, :stability, nothing, IntrinsicsWrappers.neg_float, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.neg_float_fast, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.neg_int, 5),
        (false, :stability, nothing, IntrinsicsWrappers.not_int, 5),
        (false, :stability, nothing, IntrinsicsWrappers.or_int, 5, 5),
        (true, :stability, nothing, IntrinsicsWrappers.pointerref, CoDual(p, dp), 2, 1),
        (true, :stability, nothing, IntrinsicsWrappers.pointerref, CoDual(q, dq), 2, 1),
        (true, :stability, nothing, IntrinsicsWrappers.pointerset, CoDual(p, dp), 5.0, 2, 1),
        (true, :stability, nothing, IntrinsicsWrappers.pointerset, CoDual(q, dq), 1, 2, 1),
        # rem_float -- untested and unimplemented because seemingly unused on master
        # rem_float_fast -- untested and unimplemented because seemingly unused on master
        (false, :stability, nothing, IntrinsicsWrappers.rint_llvm, 5),
        (false, :stability, nothing, IntrinsicsWrappers.sdiv_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.sext_int, Int64, Int32(1308622848)),
        (false, :stability, nothing, IntrinsicsWrappers.shl_int, 1308622848, 0xffffffffffffffe8),
        (false, :stability, nothing, IntrinsicsWrappers.sitofp, Float64, 0),
        (false, :stability, nothing, IntrinsicsWrappers.sle_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.slt_int, 4, 5),
        (false, :stability, nothing, IntrinsicsWrappers.sqrt_llvm, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.sqrt_llvm_fast, 5.0),
        (false, :stability, nothing, IntrinsicsWrappers.srem_int, 4, 1),
        (false, :stability, nothing, IntrinsicsWrappers.sub_float, 4.0, 1.0),
        (false, :stability, nothing, IntrinsicsWrappers.sub_float_fast, 4.0, 1.0),
        (false, :stability, nothing, IntrinsicsWrappers.sub_int, 4, 1),
        (false, :stability, nothing, IntrinsicsWrappers.trunc_int, UInt8, 78),
        (false, :stability, nothing, IntrinsicsWrappers.trunc_llvm, 5.1),
        (false, :stability, nothing, IntrinsicsWrappers.udiv_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.uitofp, Float16, 4),
        (false, :stability, nothing, IntrinsicsWrappers.ule_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.ult_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.urem_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.xor_int, 5, 4),
        (false, :stability, nothing, IntrinsicsWrappers.zext_int, Int64, 0xffffffff),

        # Non-intrinsic built-ins:
        # Core._abstracttype -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, nothing, __vec_to_tuple, [1.0]),
        (false, :none, nothing, __vec_to_tuple, Any[1.0]),
        (false, :none, nothing, __vec_to_tuple, Any[[1.0]]),
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
        (false, :stability, nothing, <:, Float64, Int),
        (false, :stability, nothing, <:, Any, Float64),
        (false, :stability, nothing, <:, Float64, Any),
        (false, :stability, nothing, ===, 5.0, 4.0),
        (false, :stability, nothing, ===, 5.0, randn(5)),
        (false, :stability, nothing, ===, randn(5), randn(3)),
        (false, :stability, nothing, ===, 5.0, 5.0),
        (true, :stability, nothing, Core._typevar, :T, Union{}, Any),
        (false, :none, (lb=1e-3, ub=100.0), Core.apply_type, Vector, Float64),
        (false, :none, (lb=1e-3, ub=100.0), Core.apply_type, Array, Float64, 2),
        (false, :stability, nothing, Base.arrayref, true, randn(5), 1),
        (false, :stability, nothing, Base.arrayref, false, randn(4), 1),
        (false, :stability, nothing, Base.arrayref, true, randn(5, 4), 1, 1),
        (false, :stability, nothing, Base.arrayref, false, randn(5, 4), 5, 4),
        (false, :stability, nothing, Base.arrayref, true, randn(5, 4), 1),
        (false, :stability, nothing, Base.arrayref, false, randn(5, 4), 5),
        (false, :stability, nothing, Base.arrayref, false, [1, 2, 3], 1),
        (false, :stability, nothing, Base.arrayset, false, [1, 2, 3], 4, 2),
        (false, :stability, nothing, Base.arrayset, false, randn(5), 4.0, 3),
        (false, :stability, nothing, Base.arrayset, false, randn(5, 4), 3.0, 1, 3),
        (false, :stability, nothing, Base.arrayset, true, randn(5), 4.0, 3),
        (false, :stability, nothing, Base.arrayset, true, randn(5, 4), 3.0, 1, 3),
        (false, :stability, nothing, Base.arrayset, false, [randn(3) for _ in 1:5], randn(4), 1),
        (false, :stability, nothing, Base.arrayset, false, _a, randn(4), 1),
        (false, :stability, nothing, Base.arrayset, true, [(5.0, rand(1))], (4.0, rand(1)), 1),
        (
            false,
            :stability,
            nothing,
            Base.arrayset,
            false,
            setindex!(Vector{Vector{Float64}}(undef, 3), randn(3), 1),
            randn(4),
            1,
        ),
        (
            false,
            :stability,
            nothing,
            Base.arrayset,
            false,
            setindex!(Vector{Vector{Float64}}(undef, 3), randn(3), 2),
            randn(4),
            1,
        ),
        (false, :stability, nothing, Core.arraysize, randn(5, 4, 3), 2),
        (false, :stability, nothing, Core.arraysize, randn(5, 4, 3, 2, 1), 100),
        # Core.compilerbarrier -- NEEDS IMPLEMENTING AND TESTING
        # Core.const_arrayref -- NEEDS IMPLEMENTING AND TESTING
        # Core.donotdelete -- NEEDS IMPLEMENTING AND TESTING
        # Core.finalizer -- NEEDS IMPLEMENTING AND TESTING
        # Core.get_binding_type -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, nothing, Core.ifelse, true, randn(5), 1),
        (false, :none, nothing, Core.ifelse, false, randn(5), 2),
        (false, :stability, nothing, Core.ifelse, true, 5, 4),
        (false, :stability, nothing, Core.ifelse, false, true, false),
        (false, :stability, nothing, Core.ifelse, false, 1.0, 2.0),
        (false, :stability, nothing, Core.ifelse, true, 1.0, 2.0),
        (false, :stability, nothing, Core.ifelse, false, randn(5), randn(3)),
        (false, :stability, nothing, Core.ifelse, true, randn(5), randn(3)),
        # Core.set_binding_type! -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, Core.sizeof, Float64),
        (false, :stability, nothing, Core.sizeof, randn(5)),
        # Core.svec -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, applicable, sin, Float64),
        (false, :stability, nothing, applicable, sin, Type),
        (false, :stability, nothing, applicable, +, Type, Float64),
        (false, :stability, nothing, applicable, +, Float64, Float64),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, StructFoo, :a),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, StructFoo, :b),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, MutableFoo, :a),
        (false, :stability, (lb=1e-3, ub=20.0), fieldtype, MutableFoo, :b),
        (true, :none, _range, getfield, StructFoo(5.0), :a),
        (false, :none, _range, getfield, StructFoo(5.0, randn(5)), :a),
        (false, :none, _range, getfield, StructFoo(5.0, randn(5)), :b),
        (true, :none, _range, getfield, StructFoo(5.0), 1),
        (false, :none, _range, getfield, StructFoo(5.0, randn(5)), 1),
        (false, :none, _range, getfield, StructFoo(5.0, randn(5)), 2),
        (true, :none, _range, getfield, MutableFoo(5.0), :a),
        (false, :none, _range, getfield, MutableFoo(5.0, randn(5)), :b),
        (false, :stability_and_allocs, nothing, getfield, UnitRange{Int}(5:9), :start),
        (false, :stability_and_allocs, nothing, getfield, UnitRange{Int}(5:9), :stop),
        (false, :stability_and_allocs, nothing, getfield, (5.0, ), 1),
        (false, :stability_and_allocs, nothing, getfield, (5.0, 4.0), 1),
        (false, :stability_and_allocs, nothing, getfield, (5.0, ), 1, false),
        (false, :stability_and_allocs, nothing, getfield, (5.0, 4.0), 1, false),
        (false, :stability_and_allocs, nothing, getfield, (1, ), 1, false),
        (false, :stability_and_allocs, nothing, getfield, (1, 2), 1),
        (false, :stability_and_allocs, nothing, getfield, (a=5, b=4), 1),
        (false, :stability_and_allocs, nothing, getfield, (a=5, b=4), 2),
        (false, :none, nothing, getfield, (Float64, Float64), 1),
        (false, :none, nothing, getfield, (Float64, Float64), 2, false),
        (false, :none, _range, getfield, (a=5.0, b=4), 1),
        (false, :none, _range, getfield, (a=5.0, b=4), 2),
        (false, :none, _range, getfield, UInt8, :name),
        (false, :none, _range, getfield, UInt8, :super),
        (true, :none, _range, getfield, UInt8, :layout),
        (false, :none, _range, getfield, UInt8, :hash),
        (false, :none, _range, getfield, UInt8, :flags),
        # getglobal requires compositional testing, because you can't deepcopy a module
        # invoke -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, isa, 5.0, Float64),
        (false, :stability, nothing, isa, 1, Float64),
        (false, :stability, nothing, isdefined, MutableFoo(5.0, randn(5)), :sim),
        (false, :stability, nothing, isdefined, MutableFoo(5.0, randn(5)), :a),
        # modifyfield! -- NEEDS IMPLEMENTING AND TESTING
        (false, :stability, nothing, nfields, MutableFoo),
        (false, :stability, nothing, nfields, StructFoo),
        # replacefield! -- NEEDS IMPLEMENTING AND TESTING
        (false, :none, _range, setfield!, MutableFoo(5.0, randn(5)), :a, 4.0),
        (false, :none, nothing, setfield!, MutableFoo(5.0, randn(5)), :b, randn(5)),
        (false, :none, _range, setfield!, MutableFoo(5.0, randn(5)), 1, 4.0),
        (false, :none, _range, setfield!, MutableFoo(5.0, randn(5)), 2, randn(5)),
        (false, :none, _range, setfield!, NonDifferentiableFoo(5, false), 1, 4),
        (false, :none, _range, setfield!, NonDifferentiableFoo(5, true), 2, false),
        # swapfield! -- NEEDS IMPLEMENTING AND TESTING
        # throw -- NEEDS IMPLEMENTING AND TESTING
        [false, :stability_and_allocs, nothing, tuple, 5.0, 4.0],
        [false, :stability_and_allocs, nothing, tuple, randn(5), 5.0],
        [false, :stability_and_allocs, nothing, tuple, randn(5), randn(4)],
        [false, :stability_and_allocs, nothing, tuple, 5.0, randn(1)],
        (false, :stability_and_allocs, nothing, tuple),
        (false, :stability_and_allocs, nothing, tuple, 1),
        (false, :stability_and_allocs, nothing, tuple, 1, 5),
        (false, :stability_and_allocs, nothing, tuple, 1.0, (5, )),
        [false, :stability, nothing, typeassert, 5.0, Float64],
        [false, :stability, nothing, typeassert, randn(5), Vector{Float64}],
        [false, :stability, nothing, typeof, 5.0],
        [false, :stability, nothing, typeof, randn(5)],
    ]
    memory = Any[_x, _dx, _a, p, dp]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:builtins})
    test_cases = Any[
        (false, :none, nothing, Core._apply_iterate, Base.iterate, *, 5.0, 4.0),
        (false, :none, nothing, Core._apply_iterate, Base.iterate, *, (5.0, 4.0)),
        (false, :none, nothing, Core._apply_iterate, Base.iterate, *, [5.0, 4.0]),
        (false, :none, nothing, Core._apply_iterate, Base.iterate, *, [5.0], (4.0, )),
        (false, :none, nothing, Core._apply_iterate, Base.iterate, *, 3, (4.0, )),
        (
            false, :none, nothing,
            (
                function (x)
                    rx = Ref(x)
                    pointerref(bitcast(Ptr{Float64}, pointer_from_objref(rx)), 1, 1)
                end
            ),
            5.0,
        ),
        (
            false, :none, nothing,
            (v, x) -> (pointerset(pointer(x), v, 2, 1); x), 3.0, randn(5),
        ),
        (
            false, :none, nothing,
            x -> (pointerset(pointer(x), UInt8(3), 2, 1); x), rand(UInt8, 5),
        ),
        (false, :none, nothing, getindex, randn(5), [1, 1]),
        (false, :none, nothing, getindex, randn(5), [1, 2, 2]),
        (false, :none, nothing, setindex!, randn(5), [4.0, 5.0], [1, 1]),
        (false, :none, nothing, setindex!, randn(5), [4.0, 5.0, 6.0], [1, 2, 2]),
    ]
    memory = Any[]
    return test_cases, memory
end
