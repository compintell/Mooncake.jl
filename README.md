# Taped

[![Build Status](https://github.com/withbayes/Taped.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/withbayes/Taped.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

# Getting this to work:

This currently depends on un-merged changes to Umlaut.
Please dev [my fork](https://github.com/willtebbutt/Umlaut.jl) of Umlaut and checkout to the wct/optimised_ir_tracing branch in order to get the tests to run.

# Known Limitations:

- If a (mutable) data structure `x` contains a circular reference, it will not be possible to construct a `zero_tangent` / `random_tangent` `MutableTangent` to it -- an infinite recursion will occur. It should be possible, however, to differentiate through its construction. If you find this to be a problem in practice, please open an issue.
- `zero_tangent` and `random_tangent` do not work for pointers, because we don't know how large a chunk of memory a given pointer points to, so cannot allocate a corresponding chunk of shadow memory. Your best bet when testing `rrule!!`s for things involving pointers is currently to do integration testing. See the tests for blas functionality for examples.
- If you pass active data through a global variable, AD will fail. Furthermore / worse still, the failures will probably be silent.
