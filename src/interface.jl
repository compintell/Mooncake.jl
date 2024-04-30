"""
    __value_and_pullback!!(rule, ȳ, f::CoDual, x::CoDual...)

In-place version of `value_and_pullback!!` in which the arguments have been wrapped in
`CoDual`s. Note that any mutable data in `f` and `x` will be incremented in-place. As such,
if calling this function multiple times with different values of `x`, should be careful to
ensure that you zero-out the tangent fields of `x` each time.
"""
function __value_and_pullback!!(rule::R, ȳ::T, fx::Vararg{CoDual, N}) where {R, N, T}
    out, pb!! = rule(tuple_map(to_fwds, fx)...)
    @assert _typeof(tangent(out)) == fdata_type(T)
    increment!!(tangent(out), fdata(ȳ))
    v = copy(primal(out))
    return v, tuple_map((f, r) -> tangent(fdata(tangent(f)), r), fx, pb!!(rdata(ȳ)))
end

"""
    __value_and_gradient!!(rule, f::CoDual, x::CoDual...)

Equivalent to `value_and_pullback(rule, 1.0, f, x...)` -- assumes `f` returns a `Float64`.
"""
function __value_and_gradient!!(rule::R, fx::Vararg{CoDual, N}) where {R, N}
    return __value_and_pullback!!(rule, 1.0, fx...)
end

"""
    value_and_pullback!!(rule, ȳ, f, x...)

Compute the value and pullback of `f(x...)`. `ȳ` must be a valid tangent for the primal
return by `f(x...)`.

`rule` should be constructed using `build_rrule`.

*Note:* If calling `value_and_pullback!!` multiple times for various values of `x`, you
should use the same instance of `rule` each time.

*Note:* It is your responsibility to ensure that there is no aliasing in `f` and `x`.
For example,
```julia
X = randn(5, 5)
rule = build_rrule(dot, X, X)
value_and_pullback!!(rule, 1.0, dot, X, X)
```
will yield the wrong result.

*Note:* This method of `value_and_pullback!!` has to first call `zero_codual` on all of its
arguments. This may cause some additional allocations. If this is a problem in your
use-case, consider pre-allocating the `CoDual`s and calling the other method of this
function. The `CoDual`s should be primal-tangent pairs (as opposed to primal-fdata pairs).
"""
function value_and_pullback!!(rule::R, ȳ, fx::Vararg{Any, N}) where {R, N}
    return __value_and_pullback!!(rule, ȳ, tuple_map(zero_codual, fx)...)
end

"""
    value_and_gradient!!(rule, f, x...)

Equivalent to `value_and_pullback(rule, 1.0, f, x...)` -- assumes `f` returns a `Float64`.
"""
function value_and_gradient!!(rule::R, fx::Vararg{Any, N}) where {R, N}
    return __value_and_gradient!!(rule, tuple_map(zero_codual, fx)...)
end
