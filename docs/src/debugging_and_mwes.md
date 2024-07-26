# Debugging and MWEs

There's a reasonable chance that you'll run into an issue with Tapir.jl at some point.
This page documents what you should do when this happens.

In order to produce MWEs, we recommend using `Tapir.TestUtils.test_derived_rule` to generate your test cases.

```@docs
Tapir.TestUtils.test_derived_rrule
```

## AD Doesn't Run




## AD Yields Incorrect Results



## AD Is Slow





## Segfaults

These _really_ shouldn't happen.
However, if you encounter a segfault, please re-run your problem with "safe mode" switched on (a better name for this mode might be "debug mode").
See [Safe Mode](@ref) for more info.
In general, this will catch problems before they become segfaults, at which point you can refer to the other guidance in this document.
