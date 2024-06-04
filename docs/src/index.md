# Tapir.jl

Documentation for Tapir.jl is on it's way!

Note (29/05/2024): I (Will) am currently actively working on the documentation.
It will be merged in chunks over the next month or so as good first drafts of sections are completed.
Please don't be alarmed that not all of it is here!

# Tapir.jl's Reverse-Mode Interface

The point of Tapir.jl is to perform reverse-mode algorithmic differentiation (AD).
Consequently, it must be possible to interpret Tapir.jl as computing vector-Jacobian products (VJPs).
The purpose of this section is to precisely specify _what_ Tapir.jl does, and _how_ it can be interpreted as computing VJPs.
We do this in two parts:
1. specify what types are used to represent "vector"s -- the inputs and outputs of a VJP -- and the mathematical model associated to these,
1. what the functions which compute VJPs must do, and the mathematical model associated to these.


## Tangents

Tangents (also cotangents) are what is used to represent the user-facing result of AD.
They are the V in VJP.
The extended docstring for [`tangent_type`](@ref) provides the best introduction to the types which are used to represent tangents.

```@docs
tangent_type(P)
```



## FData and RData

While tangents are the things used to represent gradients, they are not strictly what gets propagated forwards and backwards by rules during AD.
Rather, they are split into fdata and rdata, and these are passed around.

```@docs
Tapir.fdata_type(T)
```

### Why Uniqueness of Tangents / FData / RData?

Why does Tapir.jl insist that each primal type `P` be paired with a single tangent type `T`, as opposed to being more permissive.
There are a few notable reasons:
1. To provide a precise interface. Rules pass fdata around on the forwards-pass and rdata on the reverse-pass -- being able to make strong assumptions about the type of the fdata / rdata given the primal type makes implementing rules much easier in practice.
1. Conditional type stability. We wish to have a high degree of confidence that if the primal code is type-stable, then the AD code will also be. It is straightforward to construct type stable primal codes which have type-unstable forwards- and reverse-passes if you permit there to be more than one fdata / rdata type for a given primal. So while uniqueness is certainly not sufficient on its own to guarantee conditional type stability, it is probably necessary in general.
1. Test-case generation and coverage. There being a unique tangent / fdata / rdata type for each primal makes being confident that a given rule is being tested thoroughly much easier. For a given primal, rather than there being many possible input / output types to consider, there is just one.

This topic, in particular what goes wrong with permissive tangent type systems like those employed by ChainRules, deserves a more thorough treatment -- hopefully someone will write something more expansive on this topic at some point.



## The Rule Abstraction

Tapir.jl makes use of a rule system which is at first glance similar to the `rrule` function offered by ChainRules.jl.
However, owing to Tapir.jl's support for mutation (e.g. differentiating through functions which write to arrays) and high degree of precision around the types used to represent (co)-tangent-like data, the number of situations in which the two are identical are actually rather small.

Nevertheless, we begin this explanation with an example which should be familiar to anyone who has used ChainRules.jl and seen its rrule.
Once this example has been explained, we move into new territory.

### Functions of Scalars: from ChainRules.rrule to Tapir.rrule!!

Consider the simple Julia function
```julia
mul(a::Float64, b::Float64) = a * b
```

A `ChainRules.rrule` for this might look something like
```julia
function ChainRules.rrule(::typeof(mul), a::Float64, b::Float64)
    mul_pullback(dc::Float64) = NoTangent(), dc * b, dc * a
    return a * b, mul_pullback
end
```

The corresponding `Tapir.rrule!!` would be something like
```julia
function Tapir.rrule!!(::CoDual{typeof(mul)}, a::CoDual{Float64}, b::CoDual{Float64})
    _a = primal(a)
    _b = primal(b)
    mul_pullback!!(dc::Float64) = NoRData(), dc * _b, dc * _a
    return CoDual(_a * _b, NoFData()), mul_pullback!
end
```

The core differences between the `rrule` and `rrule!!` are:
1. each argument is a `CoDual`, which contains the primal and one other piece of data (more on this later),
1. we must extract the primal values from `a` and `b` using the `primal` function in order to access them,
1. `NoTangent()` is replaced by `NoRData()`, and
1. we must return another `CoDual`, rather than just the primal value (more on this later).

The point of this example is to highlight that `Tapir.rrule!!`s look a lot like `ChainRules.rrule`s in some situations, so some of your existing knowledge should transfer over.

### Functions of Vectors

We now turn to the obvious question: why do `Tapir.rrule!!`s differ from `ChainRules.rrule`s?
The short answer is that Tapir.jl requires that each unique primal memory address associated to differentiable data be associated to a unique tangent (a.k.a. shadow) memory address.
(See [Why Unique Memory Address](@ref) to understand why this is necessary.)

To see how this is achieved, consider the function
```julia
function set_1!(x::Vector{Float64}, y::Float64)
    x[1] = y
    return x
end
```
A valid `Tapir.rrule!!` for this function given below.
There are a lot of concepts introduced here, so you'll need to hop back and forth between this and the text below which explains everything.
```julia
function Tapir.rrule!!(
    ::CoDual{typeof(set_1!)}, x::CoDual{Vector{Float64}}, y::CoDual{Float64}
)
    # Extract the primal and "fdata" from x.
    px = primal(x)
    dx = tangent(x)

    # Store the current values.
    px_1_old = px[1]
    dx_1_old = dx[1]

    # Set x_p[1] to `y` and zero-out x_f[1].
    px[1] = primal(y)
    dx[1] = 0.0

    function set_1_pullback!!(::NoRData)

        # The (co)tangent to `y` is just the value in the first position of x_f.
        dy = dx

        # We _must_ undo any changes which occur on the forwards-pass, both to the primal
        # and the fdata (the forwards-component of the tangent).
        px[1] = px_1_old
        dx[1] = dx_1_old

        # There's nothing to propagate backwards for `f` because it's non-differentiable.
        # It has "no reverse data", hence `NoRData`.
        df = NoRData()

        # There's nothing to propagate backwards for `x`, because its tangent is entirely
        # represented by `dx` on the forwards-pass, hence `NoRData`.
        dx = NoRData()

        return df, dx, dy
    end

    # Just return x (the CoDual) -- this propagates forwards the correct unique tangent
    # memory for `x`.
    return x, set_1_pullback!!
end
```
Let's unpack the above:

#### Memory Propagation

We stated at the top of this section that each unique address associated to differentiable data must have a unique tangent memory address associated to it.
To see how this rule preserves this, consider the function
```julia
g(x::Vector{Float64}, y::Float64) = x, set_1!(x, y)
```
The output of `g` is a `Tuple` with the same `Vector{Float64}` in each element.
Therefore, during AD, they _must_ be associated to the same tangent address.
Happily, simple by by returning `x` at the end of the `rrule!!` for `set_1!` we ensure that this happens.

#### The other field in a `CoDual`

In this example, the other field in the `CoDual` associated to `x` must contain a `Vector{Float64}`, which represents the tangent to `x`.
We call this the _fdata_ ("forwards data") associated to `x`.
We didn't show it, but the fdata associated to `y` is `NoFData` ("no forwards data"), indicating that there is no additional data associated to `y` on the forwards-pass.

Why is this the case?


#### Summary

Note that this very simple function does _not_ have a meaningful `ChainRules.rrule` counterpart because it mutates (modifies) `x`, and `ChainRules.rrule` does not support mutation.





### The General Case
TODO


# Aside

### Why Unique Memory Address
TODO
