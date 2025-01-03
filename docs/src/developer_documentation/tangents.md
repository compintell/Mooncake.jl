# Tangents

As discussed in [Representing Gradients](@ref), Mooncake requires that each "primal" type `P` be associated to a unique "tangent" type `T`, given by the function [tangent_type](@ref).
Moreover, we must be able to "split" a given tangent into its _fdata_ ("forwards-data") and _rdata_ ("reverse-data"), whose types are given by [fdata_type](@ref) and `rdata_type` respectively.

Additionally, there is a range of things that one must be able to do with any given tangent / fdata / rdata for Mooncake to operate correctly.

