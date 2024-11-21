# Forwards-Mode Design

The purpose of this document is to explain how forwards-mode AD in Mooncake.jl is implemented.
It takes a very similar approach to reverse-mode AD, but is substantially similiar.
We discuss the notable distinctions in implementation between `Mooncake.jl`'s forwards-mode and reverse-mode later on.

## Overall Plan

This document
1. specifies the semantics of a "rule" for forwards-mode AD,
1. specifies how to implement rules by-hand for primitives, and
1. specifies how to derive rules from `IRCode` algorithmically in general.

## Forwards-Rule Semantics

Loosely, a rule for a function simultaneously
1. performs same computation as the original function, and
1. computes the Frechet derivative.

This is best made concrete through a worked example.
A rule for
```julia
z = f(x, y)
```
is a function, call it `rule_for_f`, such that
```julia
z_dz = rule_for_f(Dual(f, df), Dual(x, dx), Dual(y, dy))::Dual
```
where:
1. `rule_for_f` is a callable. It might be written by-hand, or derived algorithmically.
1. `df`, `dx`, and `dy` are tangents for `f`, `x`, and `y` respectively.
1. `z_dz` is a `Dual` containing the primal and the result of applying the derivative at `(f, x, y)` to `(df, dx, dy)` -- see see [Algorithmic Differentiation](@ref) for the definition of the derivative we use here.
1. running `rule_for_f` leaves `f`, `x`, and `y` in the same state that running `f` does.

### Tangent Types

We will use the type system documented in [Representing Gradients](@ref).
This means that every primal type has a unique tangent type.
Moreover, if a `Dual` is defined as follows:
```julia
struct Dual{P, T}
    primal::P
    tangent::T
end
```
it must always hold that `T = tangent_type(P)`.

## Hand-Written Rules

Hand-written rules are implemented by writing methods of two functions: `is_primitive` and `frule!!`.

### `is_primitive`

`is_primitive(::Type{<:Union{MinimalForwardsCtx, DefaultForwardsCtx}}, signature::Type{<:Tuple})` should return `true` if AD must attempt to differentiate a call by passing the arguments to `frule!!`, and `false` otherwise.
The [`Mooncake.@is_primitive`](@ref) macro can be used to implement this straightforwardly.

### `frule!!`

Methods of `frule!!` do the actual differentiation.

In what follows, we will refer to `frule!!`s for signatures.
For example, the `frule!!` for signature `Tuple{typeof(sin), Float64}` is the rule which would differentiate calls like `sin(5.0)`.

#### Simple Scalar Function

Recall that for ``y = \sin(x)`` we have that ``\dot{y} = \cos(x) \dot{x}``.
So the `frule!!` for signature `Tuple{typeof(sin), Float64}` is:
```julia
function frule!!(::Dual{typeof(sin)}, x::Dual{Float64})
    return Dual(sin(x.primal), cos(x.primal) * x.tangent)
end
```

#### Pre-allocated Matrix-Matrix Multiply

Recall that for ``Z = X Y`` we have that ``\dot{Z} = X \dot{Y} + \dot{X} Y``.
So the `frule!!` for signature `Tuple{typeof(mul!), Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}` is:
```julia
function frule!!(
    ::Dual{typeof(LinearAlgebra.mul!)}, z::Dual{P}, x::Dual{P}, y::Dual{P}
) where {P<:Matrix{Float64}}

    # Primal computation.
    mul!(z.primal, x.primal, y.primal)

    # Overwrite tangent of `z` to contain propagated tangent.
    mul!(z.tangent, x.primal, y.tangent)

    # Add the result of x.tangent * y.primal to `z.tangent`.
    mul!(z.tangent, x.tangent, y.primal, 0.0, 1.0) 
    return z
end
```


## Derived Rules

This is where the action takes place.
Roughly speaking, derivation of a rule for signature `sig = Tuple{F, X1, X2...}` proceeds as follows:
1. look up the optimised `Compiler.IRCode` associated to `sig`.
2. For each statement in `IRCode`, transform it according to a set of rules (listed below) to produce a new `IRCode`.
3. Apply standard Julia optimisations to this new `IRCode` (type inference, inlining, scalar reduction of aggregates, dead code elimination, etc).
4. Put this code inside a `Core.OpaqueClosure` in order to produce a executable object.
5. Put this `Core.OpaqueClosure` inside a `DerivedFRule` object to handle various bits of bookkeeping around varargs.


In order:

#### Looking up the `Compiler.IRCode`.

This is done using `Mooncake.lookup_ir`.
This function has methods with will return the `IRCode` associated to:
1. signatures (e.g. `Tuple{typeof(f), Float64}`)
1. `Base.MethodInstances` (relevant for `:invoke` expressions -- see [Statement Transformation](@ref) below)
1. `MistyClosures.MistyClosure` objects, which is essential when computing higher order derivatives and Hessians by applying `Mooncake.jl` to itself.

#### Statement Transformation

Each statment which can appear in the Julia IR is transformed by a method of `Mooncake.make_fwds_ad_stmts`.
While readers should simply refer to the methods of this function for the definitive explanation of what transformation is applied, and the rationale for applying it, we provide a high-level summary here.

There are many types of statements which can appear in `IRCode`, but the most important five are:
