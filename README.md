# Taped

[![Build Status](https://github.com/withbayes/Taped.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/withbayes/Taped.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)


# Anatomy of an Umlaut op

There are three types of operation that can live on an Umlaut tape:
1. Input
1. Constant
1. Call

There's also the un-exported `Loop`, but I don't believe that this is used anywhere currently, so we don't need to worry about it.

## `Call` anatomy

val = fn(args...)
