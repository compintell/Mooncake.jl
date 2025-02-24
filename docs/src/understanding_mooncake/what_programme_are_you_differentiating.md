# Towards AD in Julia: Composition of Rules

In [Mooncake.jl's Rule System](@ref) we provide a mathematical model for a _single_ Julia `function`, and state what a rule to differentiate it in reverse-mode must do.
However, we do not explain how to algorithmically derive rules for compositions of Julia `function`s, each of which already has a rule.
The purpose of this section is to move towards this.

## A Motivating Example

By the end of this section we will understand why, for the following function:
```julia
function f(x, y)
    a = g(x)
    b = h(a, y)
    return b
end
```
the following rule is a correct implementation of reverse-mode AD for it:
```julia
function r(f, x, y)
    a, adj_g = r(g, x)
    b, adj_h = r(h, a, x, y)
    function adj_f(db)
        _, da, dx, dy = adj_h(db)
        _, dx2 = adj_g(da)
        dx = dx + dx2
        return NoRData(), dx, dy
    end
    return b, adj_f
end
```
This is important because the above rule essentially does the following:
1. fowards-pass: replace calls to rules.
2. reverse-pass: run adjoints in reverse order, adding together rdata when a variable is used multiple times.

This way of writing rules is the essence of the "A" in "AD".
This page is therefore dedicated to building up to this example via a sequence of increasingly general examples.
Once we have this, extending it to a _very_ general class of Julia functions is comparatively straightforward.

We shall adopt the following approach to each problem:
1. specify class of `function`s,
2. specify class of differentiable functions used to model these `function`s,
3. specify how to find the adjoints of this differentiable model, and
4. describe a rule system which implements these adjoints.

At a high level, you can think of this approach as first "mathematising" the problem, applying the techniques developed in [Algorithmic Differentiation](@ref) to determine what it is that AD must do, and then providing an outline for implementing this model as a computer programme.

## Part 1: Simple Compositions of Pure Functions

For this class of `function`s, the translation between the Julia `function` and differentiable function used to model it is almost trivial.
We ask for patience, and promise that the modelling task will become more interesting shortly!

### `function` Class

To start with, let us consider only `function`s which are, pure (free of externally-visible side effects), unary, and don't contain any data themselves (e.g. no closures or callable `struct`s).
For example, consider:
#### `g`:
`g(x::Vector{Float64}) = 2x` TODO: pick a non-linear example!!!!

#### `h`:
`h(x::Matrix{Float64}) = sum(x)`.

#### Composition
Let `f` be the composition of `f_1, ..., f_N`, a collection of `N` Julia `function`s which are pure and unary.
This might be implemented as `f(x) := f_N ∘ ... ∘ f_1`, or perhaps
```julia
function f(x)
    x_1 = x
    x_2 = f_1(x_1)
    ...
    return f_N(x_N)
end
```
There are many ways to implement this function.

### Differentiable Model

We propose to represent any `function` `f` in this class by a differentiable function $$f : \mathcal{X} \to \mathcal{Y}$$.

#### `g`:
Let $$\mathcal{X} = \mathcal{Y} =: \mathbb{R}^D$$ where $$D$$ is `length(x)`, and $$f(x) := 2x$$.

#### `h`:
Let $$\mathcal{X} := \mathbb{R}^{P \times Q}$$, and $$\mathcal{Y} := \mathbb{R}$$, where $$P$$ and $$Q$$ are the number of rows and columns in `x`, and $$f(x) := \sum_{p,q} x_{p,q}$$.

#### Composition:
Let $$f_n : \mathcal{X}_n \to \mathcal{X}_{n+1}$$ be the differentiable model for `f_n`.
Then the differentiable model $$f : \mathcal{X} \to \mathcal{Y}$$ for `f` is $$f := f_N \circ \dots \circ f_1$$, with $$\mathcal{X} := \mathcal{X}_1$$ and $$\mathcal{Y} := \mathcal{X}_{N+1}$$.

### Adjoints of Model

You can apply the tools developed in [Algorithmic Differentiation](@ref) to figure out the adjoints of `g` and `h`.
The adjoint of `f` is also given there.
Let $$D f_n [x_n]^\ast$$ be the adjoint of the derivative of $$f_n$$ at $$x_n$$, then the adjoint of $$f$$ at $$x$$ is just
```math
D f [x]^\ast = D f_1 [x_1]^\ast \circ \dots \circ D f_N [x_N]^\ast.
```

### Rules

For this simple class of functions, a simple rule system will do.
We require that a rule for a `function` with mathematical model $$f$$ accepts the same argument as the original `function`, and returns a 2-tuple containing
1. the result of applying the `function` to its input, and
2. another function which implements[^implementing_mathematics_on_a_computer] the adjoint, i.e. $$D f [x]^\ast$$.

Given a rule for a `function` of interest, we simply run the rule, and can then apply the adjoint to any gradient vector of interest.

#### `g`:
```julia
function rrule(::typeof(g), x::Vector{Float64})
    g_adjoint(ȳ::Vector{Float64}) = 2ȳ
    return g(x), g_adjoint
end
```

#### `h`:
```julia
function rrule(::typeof(h), x::Matrix{Float64})
    h_adjoint(ȳ::Float64) = fill(ȳ, size(x))
    return h(x), h_adjoint
end
```

#### Composition:

One possible implementation for a rule for the composition of `f_1, ..., f_N` is
```julia
function rrule(::typeof(f), x)
    x_1 = x
    x_2, f_1_adjoint = rrule(f_1, x_1)
    ...
    y, f_N_adjoint = rrule(f_N, x_N)
    function f_adjoint(ȳ)
        x̄_N = f_N_adjoint(ȳ)
        ...
        x̄_1 = f_1_adjoint(x̄_2)
        x̄ = x̄_1
        return x̄
    end
    return y, f_adjoint
end
```
You should convince yourself that this does indeed return a 2-tuple satisfying the specification above.



## Part 2: Computational Graphs of Pure Functions

## Part 3: Computational Graphs of Mutating Functions

## Part 4: Computational Graphs of Mutating Functions with Aliasing


[^implementing_mathematics_on_a_computer]: put differently, suppose that someone wrote down some equations in a paper or textbook, and gave you a piece of code which they claim is an implementation of these equations (e.g. a neural network, a probabilistic model, an ODE, etc). Under what conditions would you be satisfied that the implementation was correct? We all do this in informal ways all of the time. I propose that you apply the same set of standards here: we have written down some equations for the adjoints, and are claiming that our rule system is an implementation of these. The fact that we arrived at this set of equations by modelling a computer programme is neither here nor there for this step of the process.
