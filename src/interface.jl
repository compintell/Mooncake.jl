"""
    value_and_gradient!!(rule, f::CoDual, x::CoDual...)

In-place version of `value_and_gradient!!` in which the arguments have been wrapped in
`CoDual`s. Note that any mutable data in `f` and `x` will be incremented in-place. As such,
if calling this function multiple times with different values of `x`, should be careful to
ensure that you zero-out the tangent fields of `x` each time.
"""
function value_and_gradient!!(rule::R, codual_fargs::Vararg{CoDual, N}) where {R, N}
    out, pb!! = rule(codual_fargs...)
    @assert out isa CoDual{Float64, Float64}
    return primal(out), pb!!(1.0, map(tangent, codual_fargs)...)
end

"""
    value_and_gradient!!(rule, f, args...)

Compute the value and gradient of `f(args...)`.

`rule` should be constructed using `build_rrule`.

*Note:* If calling `value_and_gradient!!` multiple times for various values of `args`, you
should use the same instance of `rule` each time.

*Note:* It is your responsibility to ensure that there is no aliasing in `f` and `args`.
For example,
```julia
X = randn(5, 5)
rule = build_rrule(dot, X, X)
value_and_gradient!!(rule, dot, X, X)
```
will yield the wrong result.

*Note:* This method of `value_and_gradient!!` has to first call `zero_codual` on all of its
arguments. This may cause some additional allocations. If this is a problem in your
use-case, consider pre-allocating the `CoDual`s and calling the other method of this
function.
"""
function value_and_gradient!!(rule::R, fargs::Vararg{Any, N}) where {R, N}
    return value_and_gradient!!(rule, map(zero_codual, fargs)...)
end
