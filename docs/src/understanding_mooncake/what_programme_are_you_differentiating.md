# Towards AD in Julia: Composition of Rules

In [Mooncake.jl's Rule System](@ref) we provide a mathematical model for a _single_ Julia `function`, and state what a rule to differentiate it in reverse-mode must do.
However, we do not explain how algorithmically derive rules for compositions of Julia `function`s, each of which already has a rule.
The purpose of this section is to move towards this.

## A Motivating Example

By the end of this section, we will understand how to differentiate the following Julia function:
```julia
function f(x, y)
    a = g(x)
    b = h(a, y)
    return b
end
```

We claim that the following is a valid rule for `f`:
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
i.e.
1. fowards-pass: replace calls to rules.
2. reverse-pass: run adjoints in reverse order, adding together rdata when a variable is used multiple times.

Once we understand why this is a correct way to differentiate this Julia function, extending it to a _very_ general class of Julia functions is comparatively straightforward.
We shall build towards an explanation via a sequence of even simpler problems.

We shall adopt the following approach to each problem:
1. specify class of `function`s,
2. specify class of differentiable functions used to model these `function`s,
3. specify how to find the adjoints of this differentiable model, and
4. describe a rule system which implements these adjoints.

At a high level, you can think of this approach as first "mathematising" the problem, applying the techniques developed in [Algorithmic Differentiation](@ref) to determine what it is that AD must do, and then providing an outline for implementing this model as a computer programme.

## Part 1: Simple Compositions of Pure Functions

### `function` Class

To start with, let us consider only
1. unary functions, and
2. pure functions.

For example,
```julia
g(x::Array{Float64}) = 2x
```
or
```julia
h(x::Array{Float64}) = sum(x)
```

### Differentiable Model

We propose to represent any `function` `f` in this class by a differentiable function $$f : \mathcal{X} \to \mathcal{Y}$$.



## Part 2: Computational Graphs of Pure Functions

## Part 3: Computational Graphs of Mutating Functions

## Part 4: Computational Graphs of Mutating Functions with Aliasing
