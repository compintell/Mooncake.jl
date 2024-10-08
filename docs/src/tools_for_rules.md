# Tools for Rules

Most of the time, Mooncake.jl can just differentiate your code, but you will need to intervene if you make use of a language feature which is unsupported.
However, this does not always necessitate writing your own `rrule!!` from scratch.
In this section, we detail some useful strategies which can help you avoid having to write `rrule!!`s in many situations.

## Simplfiying Code via Overlays

```@docs
Mooncake.@mooncake_overlay
```

## Functions with Zero Adjoint

If the above strategy does not work, but you find yourself in the surprisingly common
situation that the adjoint of the derivative of your function is always zero, you can very
straightforwardly write a rule by making use of the following:
```@docs
Mooncake.@zero_adjoint
Mooncake.zero_adjoint
```

## Using ChainRules.jl

[ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) provides a large number of rules for differentiating functions in reverse-mode.
These rules are methods of the `ChainRulesCore.rrule` function.
There are some instances where it is most convenient to implement a `Mooncake.rrule!!` by wrapping an existing `ChainRulesCore.rrule`.

There is enough similarity between these two systems that most of the boilerplate code can be avoided.

```@docs
Mooncake.@from_rrule
```
