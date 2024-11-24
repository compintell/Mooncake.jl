@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp_fast),IEEEFloat}
function rrule!!(
    ::CoDual{typeof(Base.FastMath.exp_fast)}, x::CoDual{P}
) where {P<:IEEEFloat}
    yp = Base.FastMath.exp_fast(primal(x))
    exp_fast_pb!!(dy::P) = NoRData(), dy * yp
    return CoDual(yp, NoFData()), exp_fast_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp2_fast),IEEEFloat}
function rrule!!(
    ::CoDual{typeof(Base.FastMath.exp2_fast)}, x::CoDual{P}
) where {P<:IEEEFloat}
    yp = Base.FastMath.exp2_fast(primal(x))
    exp2_fast_pb!!(dy::P) = NoRData(), dy * yp * log(2)
    return CoDual(yp, NoFData()), exp2_fast_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp10_fast),IEEEFloat}
function rrule!!(
    ::CoDual{typeof(Base.FastMath.exp10_fast)}, x::CoDual{P}
) where {P<:IEEEFloat}
    yp = Base.FastMath.exp10_fast(primal(x))
    exp2_fast_pb!!(dy::P) = NoRData(), dy * yp * log(10)
    return CoDual(yp, NoFData()), exp2_fast_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.sincos),IEEEFloat}
function rrule!!(::CoDual{typeof(Base.FastMath.sincos)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = Base.FastMath.sincos(primal(x))
    sincos_fast_adj!!(dy::Tuple{P,P}) = NoRData(), dy[1] * y[2] - dy[2] * y[1]
    return CoDual(y, NoFData()), sincos_fast_adj!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.log),Union{IEEEFloat,Int}}
@zero_adjoint MinimalCtx Tuple{typeof(log),Int}

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:fastmath})
    test_cases = Any[
        (false, :stability_and_allocs, nothing, Base.FastMath.exp10_fast, 0.5),
        (false, :stability_and_allocs, nothing, Base.FastMath.exp2_fast, 0.5),
        (false, :stability_and_allocs, nothing, Base.FastMath.exp_fast, 5.0),
        (false, :stability_and_allocs, nothing, Base.FastMath.sincos, 3.0),
    ]
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:fastmath})
    test_cases = Any[
        (false, :allocs, nothing, Base.FastMath.abs2_fast, -5.0),
        (false, :allocs, nothing, Base.FastMath.abs_fast, 5.0),
        (false, :allocs, nothing, Base.FastMath.acos_fast, 0.5),
        (false, :allocs, nothing, Base.FastMath.acosh_fast, 1.2),
        (false, :allocs, nothing, Base.FastMath.add_fast, 1.0, 2.0),
        (false, :allocs, nothing, Base.FastMath.angle_fast, 0.5),
        (false, :allocs, nothing, Base.FastMath.asin_fast, 0.5),
        (false, :allocs, nothing, Base.FastMath.asinh_fast, 1.3),
        (false, :allocs, nothing, Base.FastMath.atan_fast, 5.4),
        (false, :allocs, nothing, Base.FastMath.atanh_fast, 0.5),
        (false, :allocs, nothing, Base.FastMath.cbrt_fast, 0.4),
        (false, :allocs, nothing, Base.FastMath.cis_fast, 0.5),
        (false, :allocs, nothing, Base.FastMath.cmp_fast, 0.5, 0.4),
        (false, :allocs, nothing, Base.FastMath.conj_fast, 0.4),
        (false, :allocs, nothing, Base.FastMath.conj_fast, ComplexF64(0.5, 0.4)),
        (false, :allocs, nothing, Base.FastMath.cos_fast, 0.4),
        (false, :allocs, nothing, Base.FastMath.cosh_fast, 0.3),
        (false, :allocs, nothing, Base.FastMath.div_fast, 5.0, 1.1),
        (false, :allocs, nothing, Base.FastMath.eq_fast, 5.5, 5.5),
        (false, :allocs, nothing, Base.FastMath.eq_fast, 5.5, 5.4),
        (false, :allocs, nothing, Base.FastMath.expm1_fast, 5.4),
        (false, :allocs, nothing, Base.FastMath.ge_fast, 5.0, 4.0),
        (false, :allocs, nothing, Base.FastMath.ge_fast, 4.0, 5.0),
        (false, :allocs, nothing, Base.FastMath.gt_fast, 5.0, 4.0),
        (false, :allocs, nothing, Base.FastMath.gt_fast, 4.0, 5.0),
        (false, :allocs, nothing, Base.FastMath.hypot_fast, 5.1, 3.2),
        (false, :allocs, nothing, Base.FastMath.inv_fast, 0.5),
        (false, :allocs, nothing, Base.FastMath.isfinite_fast, 5.0),
        (false, :allocs, nothing, Base.FastMath.isinf_fast, 5.0),
        (false, :allocs, nothing, Base.FastMath.isnan_fast, 5.0),
        (false, :allocs, nothing, Base.FastMath.issubnormal_fast, 0.3),
        (false, :allocs, nothing, Base.FastMath.le_fast, 0.5),
        (false, :allocs, nothing, Base.FastMath.log10_fast, 0.5),
        (false, :allocs, nothing, Base.FastMath.log1p_fast, 0.5),
        (false, :allocs, nothing, Base.FastMath.log2_fast, 0.5),
        (false, :allocs, nothing, Base.FastMath.log_fast, 0.5),
        (false, :allocs, nothing, Base.FastMath.lt_fast, 0.5, 4.0),
        (false, :allocs, nothing, Base.FastMath.lt_fast, 5.0, 0.4),
        (false, :allocs, nothing, Base.FastMath.max_fast, 5.0, 4.0),
        (
            false,
            :none,
            nothing,
            Base.FastMath.maximum!_fast,
            sin,
            [0.0, 0.0],
            [5.0 4.0; 3.0 2.0],
        ),
        (false, :allocs, nothing, Base.FastMath.maximum_fast, [5.0, 4.0, 3.0]),
        (false, :allocs, nothing, Base.FastMath.min_fast, 5.0, 4.0),
        (false, :allocs, nothing, Base.FastMath.min_fast, 4.0, 5.0),
        (
            false,
            :none,
            nothing,
            Base.FastMath.minimum!_fast,
            sin,
            [0.0, 0.0],
            [5.0 4.0; 3.0 2.0],
        ),
        (false, :allocs, nothing, Base.FastMath.minimum_fast, [5.0, 3.0, 4.0]),
        (false, :allocs, nothing, Base.FastMath.minmax_fast, 5.0, 4.0),
        (false, :allocs, nothing, Base.FastMath.mul_fast, 5.0, 4.0),
        (false, :allocs, nothing, Base.FastMath.ne_fast, 5.0, 4.0),
        (false, :allocs, nothing, Base.FastMath.pow_fast, 5.0, 2.0),
        # (:allocs, Base.FastMath.pow_fast, 5.0, 2), # errors -- ADD A RULE FOR ME!
        # (:allocs, Base.FastMath.rem_fast, 5.0, 2.0), # error -- ADD A RULE FOR ME! 
        (false, :allocs, nothing, Base.FastMath.sign_fast, 5.0),
        (false, :allocs, nothing, Base.FastMath.sign_fast, -5.0),
        (false, :allocs, nothing, Base.FastMath.sin_fast, 5.0),
        (false, :allocs, nothing, Base.FastMath.cos_fast, 4.0),
        (false, :allocs, nothing, Base.FastMath.sincos_fast, 4.0),
        (false, :allocs, nothing, Base.FastMath.sinh_fast, 5.0),
        (false, :allocs, nothing, Base.FastMath.sqrt_fast, 5.0),
        (false, :allocs, nothing, Base.FastMath.sub_fast, 5.0, 4.0),
        (false, :allocs, nothing, Base.FastMath.tan_fast, 4.0),
        (false, :allocs, nothing, Base.FastMath.tanh_fast, 0.5),
    ]
    memory = Any[]
    return test_cases, memory
end
