# Tools for Rules

```@meta
DocTestSetup = quote
    using Mooncake
end
```

Most of the time, Mooncake.jl can just differentiate your code, but you will need to intervene if you make use of a language feature which is unsupported.
However, this does not always necessitate writing your own `rrule!!` from scratch.
In this section, we detail some useful strategies which can help you avoid having to write `rrule!!`s in many situations.

## Simplfiying Code via Overlays

```@docs
Mooncake.@mooncake_overlay
```

## Functions with Zero Derivative

If the above strategy does not work, but you find yourself in the surprisingly common situation that the derivative of your function is always zero, you can very straightforwardly write a rule by making use of the following:
```@docs
Mooncake.simple_zero_adjoint
```
Suppose you have a function `foo(x, y, z)` whose derivative is zero, you would write an `rrule!!` as follows:
```julia
function Mooncake.rrule!!(f::CoDual{typeof(foo)}, x::CoDual, y::CoDual, z::CoDual)
    return Mooncake.simple_zero_adjoint(f, x, y, z)
end
```
Users of ChainRules.jl should be familiar with this functionality -- it is morally the same as `ChainRulesCore.@non_differentiable`.
This approach is utilised often in Mooncake.jl's codebase.

## Using ChainRules.jl

[ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) provides a large number of rules for differentiating functions in reverse-mode.
These rules are methods of the `ChainRulesCore.rrule` function.
There are some instances where there is it most convenient to implement a `Mooncake.rrule!!` by wrapping an existing `ChainRulesCore.rrule`.

There is enough similarity between these two systems that most of the boilerplate code can be avoided.
The docstrings below explain this functionality, and how it should / should not be used.

```@docs
Mooncake.@from_rrule
Mooncake.rrule_wrapper
```

```@meta
DocTestSetup = nothing
```
