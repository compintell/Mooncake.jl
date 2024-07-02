# Safe Mode

```@meta
DocTestSetup = quote
    using Tapir, ADTypes
end
```


_**The Problem**_

A major source of potential problems in AD systems is rules returning the wrong type of tangent / fdata / rdata for a given primal value.
For example, if someone writes a rule like
```julia
function rrule!!(::CoDual{typeof(+)}, x::CoDual{<:Real}, y::CoDual{<:Real})
    plus_reverse_pass(dz::Real) = NoRData(), dz, dz
    return zero_fcodual(primal(x) + primal(y))
end
```
and calls
```julia
rrule(zero_fcodual(+), zero_fcodual(5.0), zero_fcodual(4f0))
```
then the type of `dz` on the reverse pass will be `Float64` (assuming everything happens correctly), and this rule will return a `Float64` as the rdata for `y`.
However, the primal value of `y` is a `Float32`, so the appropriate tangent type is also `Float32`.
This error may causes the reverse pass to fail loudly, but it may fail silently.
It may cause an error much later in the reverse pass, making it hard to determine that the source of the error was the above rule.
Worst of all, in some cases it could plausibly cause a segfault, which is more-or-less the worst kind of outcome possible.


_**The Solution**_

Check that the types of the fdata / rdata associated to arguments are exactly what `tangent_type` / `fdata_type` / `rdata_type` require upon entry to / exit from rules and pullbacks.

This is implemented via `SafeRRule`:
```@docs
Tapir.SafeRRule
```

You can straightforwardly enable it when building a rule via the `safety_on` kwarg in the following:
```@docs
Tapir.build_rrule
```

When using ADTypes.jl, you can choose whether or not to use it via the `safe_mode` kwarg:
```jldoctest
julia> AutoTapir(safe_mode=false)
AutoTapir(safe_mode=false)
```

### When Should You Use Safe Mode?

Only use `safe_mode` when debugging a problem.
This is because is has substantial performance implications.


```@meta
DocTestSetup = nothing
```