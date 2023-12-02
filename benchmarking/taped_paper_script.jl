using Random
import Taped: rrule!!, CoDual
using Taped.TestUtils

function f!(x::Vector{Float64}, y::Float64)
    x .*= 2y
    return x
end

function rrule!!(::CoDual{typeof(f!)}, _x::CoDual, _y::CoDual)
    x = primal(_x)
    y = primal(_y)
    x_copy = copy(x)
    x .*= 2 .* y # run primal computation
    function pb!!(_, df, dx, dy)
        x .= x_copy # restore state
        dy += 2 * sum(dx .* x) # compute cotangent w.r.t. y
        dx .= 2 .* dx .* y # calculate tangent w.r.t. x
        return df, dx, dy
    end
    return _x, pb!!
end

function rrule!!(::typeof(sin), _x::CoDual{Float64})
    y, sin_pb = rrule(sin, primal(_x))
    function sin_pb!!(dy, df, dx)
        df_inc, dx_inc = sin_pb(dy)
        return increment!!(df, df_inc), increment!!(dx, dx_inc)
    end
    return y, sin_pb!!
end

function main()
    x = randn(3)
    y = randn()
    @show x, y
    f!(x, y)
    @show x, y

    _f! = zero_codual(f!)
    _x = zero_codual(x)
    _y = zero_codual(y)

    _z, pb!! = rrule!!(_f!, _x, _y)
    tangent(_z) .= 1
    pb!!(tangent(_z), tangent(_f!), tangent(_x), tangent(_y))

    rng = Xoshiro(123456)
    TestUtils.test_rrule!!(rng, f!, x, y; perf_flag=:stability, is_primitive=false)
end
