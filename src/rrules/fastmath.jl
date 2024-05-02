@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp_fast), Float64}
function rrule!!(::CoDual{typeof(Base.FastMath.exp_fast)}, x::CoDual{Float64})
    yp = Base.FastMath.exp_fast(primal(x))
    exp_fast_pb!!(dy::Float64) = NoRData(), dy * yp
    return CoDual(yp, NoFData()), exp_fast_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp2_fast), Float64}
function rrule!!(::CoDual{typeof(Base.FastMath.exp2_fast)}, x::CoDual{Float64})
    yp = Base.FastMath.exp2_fast(primal(x))
    exp2_fast_pb!!(dy::Float64) = NoRData(), dy * yp * log(2)
    return CoDual(yp, NoFData()), exp2_fast_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp10_fast), Float64}
function rrule!!(::CoDual{typeof(Base.FastMath.exp10_fast)}, x::CoDual{Float64})
    yp = Base.FastMath.exp10_fast(primal(x))
    exp2_fast_pb!!(dy::Float64) = NoRData(), dy * yp * log(10)
    return CoDual(yp, NoFData()), exp2_fast_pb!!
end
