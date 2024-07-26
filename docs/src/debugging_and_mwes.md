# Debugging and MWEs

There's a reasonable chance that you'll run into an issue with Tapir.jl at some point.
In order to debug what is going on when this happens, or to produce an MWE, it is helpful to have a convenient way to run Tapir.jl on whatever function and arguments you have which are causing problems.

We recommend making use of Tapir.jl's testing functionality to generate your test cases:

```@docs
Tapir.TestUtils.test_rule
```

This approach is convenient because it can
1. check whether AD runs at all,
1. check whether AD produces the correct answers,
1. check whether AD is performant, and
1. can be used without having to manually generate tangents.

## Example

```@meta
DocTestSetup = quote
    using Random, Tapir
end
```

For example
```julia
f(x) = Core.bitcast(Float64, x)
Tapir.TestUtils.test_rule(
    Random.Xoshiro(123), f, 3;
    is_primitive=false, perf_flag=:none, interp=Tapir.TapirInterpreter(),
)
```
will error.
(In this particular case, it is caused by Tapir.jl preventing you from doing (potentially) unsafe casting. In this particular instance, Tapir.jl just fails to compile, but in other instances other things can happen.)

In any case, the point here is that `Tapir.TestUtils.test_rule` provides a convenient way to produce and report an error.

## Segfaults

These are everyone's least favourite kind of problem, and they should be _extremely_ rare in Tapir.jl.
However, if you are unfortunate enough to encounter one, please re-run your problem with the `safety_on` kwarg set to `true`.
See [Safe Mode](@ref) for more info.
In general, this will catch problems before they become segfaults, at which point the above strategy for debugging and error reporting should work well.
