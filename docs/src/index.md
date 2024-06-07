# Tapir.jl

Documentation for Tapir.jl is on it's way!

Note (29/05/2024): I (Will) am currently actively working on the documentation.
It will be merged in chunks over the next month or so as good first drafts of sections are completed.
Please don't be alarmed that not all of it is here!

# Tapir.jl and Reverse-Mode AD

The point of Tapir.jl is to perform reverse-mode algorithmic differentiation (AD).
The purpose of this section is to explain _what_ precisely is meant by this, and _how_ it can be interpreted mathematically.
1. we recap what AD is, and introduce the mathematics necessary to understand is,
1. explain how this mathematics relates to functions and data structures in Julia, and
1. how this is handled in Tapir.jl.

Since Tapir.jl supports in-place operations / mutation, these will push beyond what is encountered in Zygote / Diffractor / ChainRules.
Consequently, while there is a great deal of overlap with these existing systems, in order to understand Tapir.jl properly you will need to read through this section of the docs.

## Prerequisites and Resources

This introduction assumes familiarity with the differentiation of vector-valued functions -- familiarity with the gradient and Jacobian matrices is a given.

In order to provide a convenient exposition of AD, we need to abstract a little further than this and make use of a slightly more general notion of the derivative, gradient, and "transposed Jacobian".
Please note that, fortunately, we only ever have to handle finite dimensional objects when doing AD, so there is no need for any knowledge of functional analysis to understand what is going on here.
These concepts will be introduced here, but I cannot promise that these docs give the best exposition -- they're most appropriate as a refresher and to establish the notation that I'll use.
Rather, I would recommend a couple of lectures from the "Matrix Calculus for Machine Learning and Beyond" course, which you can find [on MIT's OCW website](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/), delivered by Edelman and Johnson (who will be familiar faces to anyone who has spent much time in the Julia world!).
It is designed for undergraduates, and is accessible to anyone with some undergraduate-level linear algebra and calculus.
While I recommend the whole course, Lecture 1 part 2 and Lecture 4 part 1 are especially relevant to the problems we shall discuss -- you can skip to 11:30 in Lecture 4 part 1 if you're in a hurry.



## Derivatives


The foundation on which all of AD is built the the derivate -- we need a fairly general definition of it, so we review it here.

Let ``f : \RR \to \RR`` be differentiable everywhere.
Its derivative at some point ``x \in \RR`` is the scalar ``\alpha \in \RR`` such that
```math
\text{d}f = \alpha \, \text{d}x
```
This generalises to other kinds of vector spaces.
For example, if ``f : \RR^P \to \RR^Q`` is differentiable at a point ``x \in \RR^P``, then the derivative of ``f`` at ``x`` is given by the Jacobian matrix at ``x``, which we denote as ``J[x] \in \RR^{Q \times P}`` such that
```math
\text{d}f = J[x] \, \text{d}x
```

It is possible to stop here, as all the functions we shall need to consider can in principle be written as functions on some subset ``\RR^P``.
However, writing functions in this way turns out to be incredibly inconvenient in general.
For example, consider the convolution ``W \ast X``, where ``W`` and ``X`` are matrices -- the function ``X \to W \ast X`` is plainly a finite-dimensional linear operator, so we _could_ express it as a matrix-vector product given an appropriate mapping between the matrix ``X`` and a column vector. However, this is best avoided when possible.


Instead, we consider functions ``f : \mathcal{X} \to \mathcal{Y}``, where ``\mathcal{X}`` and ``\mathcal{Y}`` are finite dimensional Hilbert spaces.
In this instance, the derivative of ``f`` at ``x \in \mathcal{X}`` is the linear operator ``D f [x]`` satisfying
```math
\text{d}f = D f [x] \, \text{d} x
```
This is a generalisation of the previous cases. For example, if it _is_ ``\mathcal{X} = \RR^P`` and ``\mathcal{Y} = \RR^Q`` then this operator can be specified in terms of the Jacobian matrix: ``D f [x] (\text{d}x) = J \text{d} x`` -- brackets are used to emphasise that ``D f [x]`` is a function, and is being applied to ``\text{d} x``.

### An aside: _what_ does Forwards-Mode AD compute?

At this point we have enough machinery to discuss forwards-mode AD.
We do this for completeness -- feel free to skip to the next section if you are not interested in this.
Expressed in the language of linear operators and Hilbert spaces, the goal of forwards-mode AD is the following:
given a function ``f`` which is differentiable at a point ``x``, compute ``D f [x] (\dot{x})`` for a given vector ``\dot{x}``.
If ``f : \RR^P \to \RR^Q``, this is equivalent to computing ``J[x] \dot{x}``, where ``J[x]`` is the Jacobian of ``f`` at ``x``.
We provide a high-level explanation of _how_ forwards-mode AD does this in [_How_ does Forwards-Mode AD work?](@ref).



## Reverse-Mode AD: _what_ does it do?

In order to explain what AD does, it's first helpful to consider it in a familiar context: Euclidean space, ``\RR^N``.
We then generalise to more general vector spaces (although nothing infinite dimensional, as promised), which is necessary for the general case.

### Reverse-Mode AD: what does it do in Euclidean space?

In this setting, the goal of reverse-mode AD is the following: given a function ``f : \RR^P \to \RR^Q`` which is differentiable at ``x \in \RR^P`` with Jacobian ``J[x]`` at ``x``, compute ``J[x]^\top \bar{y}`` for any ``\bar{y} \in \RR^Q``.
As with forwards-mode AD, this is achieved by first decomposing ``f`` into the composition ``f = f_N \circ \dots \circ f_1``, where each ``f_n`` is a simple function whose Jacobian ``J_n[x]`` we can compute at any point ``x_n``. By the chain rule, we have that
```math
J[x] = J_N[x] \dots J_1[x] .
```
Taking the transpose and multiplying from the left by ``\bar{y}`` yields
```math
J[x]^\top \bar{y} = J[x]^\top_N \dots J[x]^\top_1 \bar{y} .
```

We see that ``J[x]^\top \bar{y}`` can be evaluated from right to left as a sequence of ``N`` matrix-vector products.

### Adjoints

In order to generalise this algorithm to work with linear operators, we must generalise the idea of multiplying a vector by the transpose of the Jacobian.
The relevant concept here is that of the _adjoint_ _operator_.
Specifically, the ``A^\ast`` of linear operator ``A`` is the linear operator satisfying
```math
\langle A^\ast \bar{y}, \dot{x} \rangle = \langle \bar{y}, A \dot{x} \rangle.
```
The relationship between the adjoint and matrix transpose is this: if ``A (x) := J x`` for some matrix ``J``, then ``A^\ast (y) := J^\top y``.

Moreover, just as ``(A B)^\top = B^\top A^\top`` when ``A`` and ``B`` are matrices, ``(A B)^\ast = B^\ast A^\ast`` when ``A`` and ``B`` are linear operators.
This result follows in short order from the definition of the adjoint operator -- proving this is a good exercise!

### Reverse-Mode AD: _what_ does it do in general?

Equipped with adjoints, we can express reverse-mode AD only in terms of linear operators, dispensing with the need to express everything in terms of Jacobians.
The goal of reverse-mode AD is as follows: given a differentiable function ``f : \mathcal{X} \to \mathcal{Y}``, compute ``D f [x]^\ast (\bar{y})`` for some ``\bar{y}``.
We will explain _how_ reverse-mode AD goes about computing this after some worked examples.

### Some Worked Examples

We now present some worked examples in order to prime intuition, and to introduce the important classes of problems that will be encountered when doing AD in the Julia language.
We will put all of these problems in a single general framework later on.

#### An Example with Matrix Calculus

We have introduced some mathematical abstraction in order to simplify the calculations involved in AD.
To this end, we consider differentiating ``f(X) = X^\top X``.
Results for this and many similar operations are given by [giles2008extended](@cite).
A similar operation, but which maps from matrices to ``\RR`` is discussed in Lecture 14 part 2 of the MIT course mentioned previouly.

The derivative of this function is
```math
D f [X] (\dot{X}) = \dot{X}^\top X + X^\top \dot{X}
```
Observe that this is indeed a linear operator (i.e. it is linear in its argument, ``\dot{X}``).
You can plug it in to the definition of the Frechet derivative in order to confirm that it is indeed the derivative.

In order to perform reverse-mode AD, we need to find the adjoint operator.
Using the usual definition of the inner product between matrices,
```math
\langle X, Y \rangle := \textrm{tr} (X^\top Y)
```
we can derive the adjoint operator:
```math
\begin{align}
    \langle \bar{Y}, D f [X] (\dot{X}) \rangle &= \langle \bar{Y}, \dot{X}^\top X + X^\top \dot{X} \rangle \nonumber
        &= \langle 
\end{align}
```

#### AD with Immutables

The way that Tapir.jl handles immutable data is very similar to how Zygote / ChainRules do.
For example, consider the Julia function
```julia
f(x::Float64, y::Tuple{Float64, Float64}) = x + y[1] * y[2]
```
If you've previously worked with ChainRules / Zygote, without thinking too hard about the formalisms we introduced previously (perhaps by considering a variety of partial derivatives) you can probably arrive at the following adjoint for the derivative of `f`:
```julia
g -> (g, (y[2] * g, y[1] * g))
```

It is helpful to work through this simple example in detail, as the steps involved apply more generally.
If at any point this exercise feels pedantic, we ask you to stick with it.
The goal is to spell out the steps involved in excessive detail, as this level of detail will be required in more complicated examples, and it is most straightforwardly demonstrated in a simple case.



##### Step 1: produce a mathematical model

There are a couple of aspects of `f` which require thought:
1. it has two arguments -- we've only handled single argument functions previously, and
2. the second argument is a `Tuple` -- we've not decided how to model this.

To this end, we define a mathematical notion of a tuple.
A tuple is a collection of ``N`` elements, each of which is drawn from some set ``\mathcal{X}_n``.
We denote by ``\mathcal{X} := \{ \mathcal{X}_1 \times \dots \times \mathcal{X}_N \}`` the set of all ``N``-tuples whose ``n``th element is drawn from ``\mathcal{X}_n``.
Provided that each ``\mathcal{X}_n`` forms a finite Hilbert space, ``\mathcal{X}`` forms a Hilbert space with
1. ``\alpha x := (\alpha x_1, \dots, \alpha x_N)``,
2. ``x + y := (x_1 + y_1, \dots, x_N + y_N)``, and
3. ``\langle x, y \rangle := \sum_{n=1}^N \langle x_n, y_n \rangle``.

We can think of multi-argument functions as single-argument functions of a tuple, so a reasonable mathematical model for `f` might be a function ``f : \{ \RR \times \{ \RR \times \RR \} \} \to \RR``, where
```math
f(x, y) := x + y_1 y_2
```
Note that while the function is written with two arguments, you should treat them as a single tuple, where we've assigned the name ``x`` to the first element, and ``y`` to the second.

##### Step 2: obtain the derivative

Now that we have a mathematical object, we can differentiate it:
```math
D f [x, y](\dot{x}, \dot{y}) = \dot{x} + \dot{y}_1 y_2 + y_1 \dot{y}_2
```

##### Step 3: obtain the adjoint

``D f[x, y]`` maps ``\{ \RR \times \{ \RR \times \RR \}\}`` to ``\RR``, so ``D f [x, y]^\ast`` must map the other way.
The reader should verify that the following follows quickly from the definition of the adjoint:
```math
D f [x, y]^\ast (\bar{f}) =  (\bar{f}, (\bar{f} y_2, \bar{f} y_1))
```


#### AD with mutable data

In the previous two examples there was an obvious mathematical model for the Julia function.
Indeed this model was sufficiently obvious that it required little explanation.
This is not always the case though, in particular, Julia functions which modify / mutate their inputs require a little more thought.

Consider the following Julia `function`:
```julia
function f!(x::Vector{Float64})
    x .*= x
    return nothing
end
```
This `function` squares each element of its input in-place, and returns `nothing`.
So what is an appropriate mathematical model for this `function`?
In order to clarify the problem further, consider another `function` which makes use of `f!` internally:
```julia
function g!(x::Vector{Float64})
    f!(x)
    return sum(x)
end
```
`g!` looks a bit more like what we've seen previously in that it returns something that we can treat as a real number.
However, unlike before, the value associated to the elements of `x` changes throughout execution.

The way to handle these two difference from before is to:
1. introduce a notion of state to each variable involved in a `function`, and
2. include the arguments in the value returned by the mathematical model.

##### Step 1: produce a mathematical model

We model `square!`


## Reverse-Mode AD: _how_ does it do it?

As discussed in [Reverse-Mode AD: _what_ does it do?](@ref) the purpose of reverse-mode AD is to apply the adjoint of the derivative, ``D f (x)^\ast`` to a vector ``\bar{y}``.

Decomposing ``f`` as before, ``f = f_N \circ \dots \circ f_1``, we assume that we can compute the adjoint of the derivative of each ``f_n``.
Then the adjoint is
```math
D f [x]^\ast (\bar{y}) = (D f_1 [x_1]^\ast \circ \dots \circ D f_N [x_N]^\ast)(\bar{y})
```
Our previous reverse-mode AD algorithm can now be generalised.

Forwards-Pass:
1. ``x_1 = x``, ``n = 1``
2. construct ``D f_n [x_n]^\ast``
3. let ``x_{n+1} = f_n (x_n)``
4. let ``n = n + 1``
5. if ``n < N + 1`` then go to 2

Reverse-Pass:
1. let ``\bar{x}_{N+1} = \bar{y}``
2. let ``n = n - 1``
3. let ``\bar{x}_{n} = D f_n [x_n]^\ast (\bar{x}_{n+1})``
4. if ``n = 1`` return ``\bar{x}_1`` else go to 2.






## A Rough Mathemtical Model for a Julia Function




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


# Asides

### _How_ does Forwards-Mode AD work?

Forwards-mode AD achieves this by breaking down ``f`` into the composition ``f = f_N \circ \dots \circ f_1``, # where each ``f_n`` is a simple function whose derivative (function) ``D f_n [x_n]`` we know for any given ``x_n``. By the chain rule, we have that
```math
D f [x] (\dot{x}) = D f_N [x_N] \circ \dots \circ D f_1 [x_1] (\dot{x})
```
which suggests the following algorithm:
1. let ``x_1 = x``, ``\dot{x}_1 = \dot{x}``, and ``n = 1``
2. let ``\dot{x}_{n+1} = D f_n [x_n] (\dot{x}_n)``
3. let ``x_{n+1} = f(x_n)``
4. let ``n = n + 1``
5. if ``n = N+1`` then return `\dot{x}_{N+1}`, otherwise go to 2.

When each function ``f_n`` maps between Euclidean spaces, the applications of derivatives ``D f_n [x_n] (\dot{x}_n)`` are given by ``J_n \dot{x}_n`` where ``J_n`` is the Jacobian of ``f_n`` at ``x_n``.v

### Why Unique Memory Address
TODO


```@bibliography
```
