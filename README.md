# Taped

[![Build Status](https://github.com/withbayes/Taped.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/withbayes/Taped.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

# Project Goals

The goal of the `Taped.jl` project is to produce a reverse-mode AD package, written entirely in Julia, which improves over both ReverseDiff.jl and Zygote.jl along several axes:
1. performance,
1. correctness / scope of testing,
1. coverage of language features.

The most notable feature that we improve on over ReverseDiff and Zygote / ChainRules is support for mutation (writing to arrays, modifying fields of `mutable struct`s, etc), which is arguably the core limitation of these two packages, and has been the elephant-in-the-room of reverse-mode AD in Julia for years.

Our system is based around a single function `rrule!!`.
It should be thought of as being similar to ChainRules' `rrule` and Zygote's `_pullback`, but with additional features / requirements which make it suitable for being applied to functions which (potentially) modify their arguments.
It has, perhaps unsurprisingly, wound up looking quite similar to the rule system in Enzyme.

We view Enzyme as the most comparable system to ours, because it also supports mutation.
The core difference between our two packages is that Enzyme targets LLVM, and its backbone is therefore written in C++.
Conversely, as stated above, Taped is written entirely in Julia.
These two approaches entail different tradeoffs.

# Project Name

The package is called `Taped.jl` because we're currently using a traditional tape-based AD system.
That is, at present, we implement `rrule!!` for an arbitrary function by tracing the function's execution onto a tape -- an operation winds up on the tape if there is a method of `rrule!!` which applies to it.
However, the work is broader in scope than tape-based AD though, so this name is somewhat misleading.
Please do _not_ assume from the name that we just care about traditional "Wengert list" tape-based AD.

# Project Status

The project plan is to broadly proceed in three phases:
1. design, correctness and testing
1. performance optimisation
1. maintenance

You should take this with a pinch of salt, as it seems highly likely that we will have to revisit some design choices when optimising performance -- we do not, however, anticipate requiring major re-writes to the design as part of performance optimisation.
We aim to reach the maintenance phase of the project before 01/06/2024.

At the time of writing (06/11/2023), we are mostly through the first phase.
Correctness testing is proceeding well, and we are ironing out known issues.
Notably, our broad experience at present is that at we continue to increase the amount of Julia code on which the package is tested, things are failing for known, predictable, straightforwardly-fixable reasons, (largely missing rrules for `ccall`s) rather than unanticipating problems.

Please note that, since we have yet to enter phase 2 of the project, we have spent _no_ time whatsoever optimising for performance.
We strongly believe that there is nothing in principle preventing us from achieving excellent performance.
However, currently, you should expect to experience _amazingly_ poor performance.
