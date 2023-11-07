# Taped

[![Build Status](https://github.com/withbayes/Taped.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/withbayes/Taped.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

The goal of the `Taped.jl` project is to produce an reverse-mode AD package which is written entirely in Julia, and improves over both `ReverseDiff.jl` and `Zygote.jl` in several ways.

# How it works

`Taped.jl` is based around a single function `rrule!!`, which should be thought of as providing a way to compute vector-Jacobian products.
It should be thought of as being similar to ChainRules' `rrule` and Zygote's `_pullback`, but with additional features / requirements which make it suitable for being applied to functions which modify their arguments, in addition to those that do not.
It has, perhaps unsurprisingly, wound up looking quite similar to the rule system in Enzyme.

For a given function and arguments, it is roughly speaking the case that either
1. a hand-written method of `rrule!!` is applicable, or
2. no hand-written method of `rrule!!` is applicable, in which case we "do AD" -- we decompose the function into a composition of functions which _do_ have hand-written `rrule!!`s.

In general, the goal is to write as few hand-written `rrule!!`s as is necessary, and to "do AD" for the vast majority of functions.


# Project Goals

Below the four central objectives of this project are outlined.

### Coverage of more of the Language

The need for first-class support for mutation has been well understood in the Julia AD community for a number of years now.
Its primary benefit is the ability to differentiate through the truly vast quantity of mutating code on which users depend in `Base` / `Core`, the standard libraries, and packages in the general registry -- empowering users to AD through code which _they_ write in a mutating way is often of secondary importance.
Thus you should equate `rrule!!`s support for mutation with support for existing code.
Conversely you should equate `Zygote.jl`'s / `ReverseDiff.jl`'s patchy support for mutation with patchy support for existing code.


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

At present (11/2023), we do _not_ do this in a performant way, but this will change.
See [Project Status](#project-status) below for more info.

### Written entirely in Julia

`Taped.jl` is written entirely in Julia.
While `Enzyme.jl` is quite comparable to `Taped.jl` in that it also supports mutation, it differs greatly in that it targets LLVM, and its backbone is therefore written in C++.
These two approaches entail different tradeoffs.

# Project Name

The package is called `Taped.jl` because we're currently using a traditional tape-based AD system.
That is, at present, we implement `rrule!!` for an arbitrary function by tracing the function's execution onto a tape -- an operation winds up on the tape if there is a method of `rrule!!` which applies to it.
However, the work is broader in scope than tape-based AD though, so this name is somewhat misleading.
Please do _not_ assume from the name that we just care about traditional "Wengert list" tape-based AD.

# Project Status

The plan is to proceed in three phases:
1. design, correctness and testing
1. performance optimisation
1. maintenance

You should take this with a pinch of salt, as it seems highly likely that we will have to revisit some design choices when optimising performance -- we do not, however, anticipate requiring major re-writes to the design as part of performance optimisation.
We aim to reach the maintenance phase of the project before 01/06/2024.

At the time of writing (06/11/2023), we are mostly through the first phase.
Correctness testing is proceeding well, and we are ironing out known issues.
Notably, our broad experience at present is that as we continue to increase the amount of Julia code on which the package is tested, things fail for known, predictable, straightforwardly-fixable reasons (largely missing rrules for `ccall`s), rather than unanticipated problems.

Please note that, since we have yet to enter phase 2 of the project, we have spent _no_ time whatsoever optimising for performance.
We strongly believe that there is nothing in principle preventing us from achieving excellent performance.
However, currently, you should expect to experience _amazingly_ poor performance.
