# Tangents

As discussed in [Representing Gradients](@ref), Mooncake requires that each "primal" type be associated to a unique "tangent" type, given by the function [tangent_type](@ref).
Moreover, we must be able to "split" a given tangent into its _fdata_ ("forwards-data") and _rdata_ ("reverse-data"), whose types are given by [fdata_type](@ref) and `rdata_type` respectively.

Very occassionally it may be necessary to specify your own tangent type.
This is not an entirely trivial undertaking, as there is quite a lot of functionality that must be added to make it work properly.
So, before diving in to add your own custom type, seriously consider whether it is worth the effort, and whether the default definition given by Mooncake are really inadequate for your use-case.

## Testing Functionality

The interface is given in the form of three functions:
```@docs
Mooncake.TestUtils.test_tangent_interface
Mooncake.TestUtils.test_tangent_splitting
Mooncake.TestUtils.test_rule_and_type_interactions
```

You can call all three of these functions at once using
```@docs
Mooncake.TestUtils.test_data
```

If all the tests in these functions pass, then you have satisfied the interface.

## Interface

Below are the docstrings for each function tested by [`Mooncake.TestUtils.test_tangent_interface`](@ref) and [`Mooncake.TestUtils.test_tangent_splitting`](@ref).

```@docs
Mooncake.tangent_type
Mooncake.zero_tangent
Mooncake.randn_tangent
Mooncake.TestUtils.has_equal_data
Mooncake.increment!!
Mooncake.set_to_zero!!
Mooncake._add_to_primal
Mooncake._diff
Mooncake._dot
Mooncake._scale
Mooncake.TestUtils.populate_address_map
Mooncake.fdata_type
Mooncake.rdata_type
Mooncake.fdata
Mooncake.rdata
Mooncake.uninit_fdata
```