# Taped

[![Build Status](https://github.com/withbayes/Taped.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/withbayes/Taped.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

The goal of the `Taped.jl` project is to produce a reverse-mode AD package which is written entirely in Julia, and improves over both `ReverseDiff.jl` and `Zygote.jl` in several ways.

# How it works

`Taped.jl` is based around a single function `rrule!!`, which computes vector-Jacobian products (VJPs).
These VJPs can, for example, be used to compute gradients.
`rrule!!` is similar to ChainRules' `rrule` and Zygote's `_pullback`, but supports functions which mutate (modify) their arguments, in addition to those that do not, and immediately increments (co)tangents.
It has, perhaps unsurprisingly, wound up looking quite similar to the rule system in Enzyme.

For a given function and arguments, it is roughly speaking the case that either
1. a hand-written method of `rrule!!` is applicable, or
2. no hand-written method of `rrule!!` is applicable.

In the first case, we run the `rrule!!`.
In the second, we create an `rrule!!` by "doing AD" -- we decompose the function into a composition of functions which _do_ have hand-written `rrule!!`s.
In general, the goal is to write as few hand-written `rrule!!`s as is necessary, and to "do AD" for the vast majority of functions.


# Project Goals

Below the four central objectives of this project are outlined, and the approaches that we take to achieve them.

### Coverage of more of the Language

The need for first-class support for mutation has been well understood in the Julia AD community for a number of years now.
Its primary benefit is the ability to differentiate through the truly vast quantity of mutating code on which users depend in `Base` / `Core`, the standard libraries, and packages in the general registry -- empowering users to AD through code which _they_ write in a mutating way is often of secondary importance.
Thus you should equate `rrule!!`'s support for mutation with good support for existing code.
Conversely, you should equate `Zygote.jl`'s / `ReverseDiff.jl`'s patchy support for mutation with patchy support for existing code.

`rrule!!`s impose no constraints on the types which can be operated on, as with `ChainRules`'s `rrule` and `Zygote`'s `_pullback`.
Consequently, there is in principle nothing to prevent `Taped.jl` from operating on any type, for example, structured arrays, GPU arrays, and complicated `struct`s / `mutable struct`s.


### Correctness and Testing

Mutation support enables writing `rrule!!`s at a low-level (`Core.InstrincFunction`s, `build-in function`s, `ccall`s to e.g. `BLAS` and `LAPACK` and the bits of Base Julia which are implemented in C).
The simplicity of this low-level functionality makes implementing correct `rrule!!`s for it a simple task.
In conjunction with being strict about the types used internally to represent (co)tangents, we have found this leads to `rrule!!`s composing very well, and AD being correct in practice as a consequence.

Furthermore, the interfaces for `rrule!!` and the (co)tangent type system have been designed to make a reliable `test_rrule!!` function possible to create.
All of our testing is implemented via this (or via another function which calls this) which makes adding test-cases trivial, and has meant that we have produced a lot of test cases.

This contrasts with `Zygote.jl` / `ChainRules.jl`, where the permissive (co)tangent type system complicates both composition of `rrule`s and testing.

Additionally, we augment the tape that we construct with additional instructions which throw an error if control flow differs from when the tape was constructed.
This contrasts with `ReverseDiff.jl`, which silently fails in this scenario.

### Performance

Hand-written `rrule!!`s have excellent performance, provided that they have been written well.
Consequently, whether or not the overall AD system has good performance is largely a question of how much overhead is associated to the mechanism by which hand-written `rrules!!`s are algorithmically composed.

~~At present (11/2023), we do _not_ do this in a performant way, but this will change.~~
At present (01/2024), we do this in a _moderately_ performant way.
See [Project Status](#project-status) below for more info.

Additionally, the strategy of immediately incrementing (co)tangents resolves long-standing performance issues associated with indexing arrays.

### Written entirely in Julia

`Taped.jl` is written entirely in Julia.
This sits in contrast to `Enzyme.jl`, which targets LLVM and is primarily written in C++.
These two approaches entail different tradeoffs.

# Project Name

The package is called `Taped.jl` because it originally used a traditional tape-based AD system.
This is no longer the case, so the name is now somewhat arbitrary.
Please do _not_ assume from the name that we just care about traditional "Wengert list" tape-based AD.

# Project Status

The plan is to proceed in three phases:
1. design, correctness and testing
1. performance optimisation
1. maintenance

You should take this with a pinch of salt, as it seems highly likely that we will have to revisit some design choices when optimising performance -- we do not, however, anticipate requiring major re-writes to the design as part of performance optimisation.
We aim to reach the maintenance phase of the project before 01/06/2024.

*Update: (16/01/2023)*
Phase 2 is now well underway. We now make use of a much faster approach to interpreting / executing Julia code, which yields performance that is comparable with ReverseDiff (when things go well). The current focus is on ironing out performance issues, and simplifying the implementation.

*Update: (06/11/2023)*
We are mostly through the first phase.
Correctness testing is proceeding well, and we are ironing out known issues.
Notably, our broad experience at present is that as we continue to increase the amount of Julia code on which the package is tested, things fail for known, predictable, straightforwardly-fixable reasons (largely missing rrules for `ccall`s), rather than unanticipated problems.

Please note that, since we have yet to enter phase 2 of the project, we have spent _no_ time whatsoever optimising for performance.
We strongly believe that there is nothing in principle preventing us from achieving excellent performance.
However, currently, you should expect to experience _amazingly_ poor performance.

# Trying it out

There is not presently a high-level interface to which we are commiting, but if you want to
compute the gradient of a function, take a look at
`Taped.TestUtils.set_up_gradient_problem` and `Taped.TestUtils.value_and_gradient!!`.
