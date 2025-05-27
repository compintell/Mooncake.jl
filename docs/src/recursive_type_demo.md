
# Supporting Recursive Tangent Types in Mooncake.jl

Mooncake.jl associates each **primal type** (the original data structure) with a unique **tangent type** (the type of its derivative information). 
By default, Mooncake can automatically derive tangent types for most Julia structs. 
However, for *recursive types* (e.g., self-referential or cyclic structures), the default mechanism can fail (often via a stack overflow). 
In such cases, you need to manually define a custom tangent type and implement the required interface. 
This tutorial will guide you through the process step by step, from understanding Mooncake’s tangent design to testing your custom tangent type.

## Understanding Tangent Types and the FData/RData Split

Before diving in, let's recap how Mooncake represents gradients (tangents) and why **forward data** (fdata) and **reverse data** (rdata) are needed. For a more detailed review, see the [Mooncake.jl Rule System](https://chalk-lab.github.io/Mooncake.jl/stable/understanding_mooncake/rule_system/):

- **Tangent Types:** For a given primal type `P`, `Mooncake.tangent_type(P)` returns the type of the tangent (gradient) associated with `P`. By default, Mooncake uses generic `Tangent{...}` structs to hold fieldwise derivatives. For example, a simple struct’s tangent might be represented as `Tangent{NamedTuple}` with the same field names as the primal. Mutable structs similarly get a `MutableTangent` type. Each field’s tangent is of type `tangent_type(field_type)`.

- **Forward vs Reverse Data:** Mooncake splits a tangent object into two parts: **fdata** (for forward-pass data) and **rdata** (for reverse-pass data). Conceptually, components of the tangent that are **identified by address** (e.g., arrays or mutable fields) are carried along during the forward pass (as fdata) and updated in-place in the reverse pass. Components identified purely by their **value** (e.g., plain numbers) are only needed during the reverse pass (as rdata). This design improves performance by minimizing what needs to be propagated forward.

To illustrate, consider a composite type like `Tuple{Float64, Vector{Float64}, Int}`. Its tangent type is `Tuple{Float64, Vector{Float64}, NoTangent}` (since an `Int` is non-differentiable, it uses a `NoTangent` placeholder). The fdata type in this case would be `Tuple{NoFData, Vector{Float64}, NoFData}`—meaning only the vector (address-identified) is forwarded, while the float and int produce no forward data. The rdata type would be `Tuple{Float64, NoRData, NoRData}`, carrying the float’s derivative (and nothing for the vector or int, since the vector’s changes are handled in-place). Mooncake ensures that you can always reconstruct the full tangent from its fdata and rdata: for any tangent `t`, if `f = Mooncake.fdata(t)` and `r = Mooncake.rdata(t)`, then `Mooncake.tangent(f, r)` should return the original `t`.

With these concepts in mind, we can see why **recursive types** pose a challenge. A *recursive type* is a struct that contains itself (directly or indirectly) as a field. For example:

```@example recur_tuto
using Mooncake: Mooncake
using DifferentiationInterface
```

```@example recur_tuto
mutable struct A{T}
    x::T
    a::Union{A{T},Nothing}

    A(x::T) where {T} = new{T}(x, nothing)
    A(x::T, child::A{T}) where {T} = new{T}(x, child)
end
```

Here, `A` has a self-referential field `a`. If we naively ask Mooncake for the tangent type of `A{Float64}`, it would try to construct something like `Tangent{Tuple{Float64, tangent_type(A)}}`—which leads to infinite recursion (since `tangent_type(A)` would embed itself again). Indeed, calling `tangent_type(A)` in this scenario would overflow the stack. The solution is to manually define a custom tangent type that breaks this circular dependency.

## Defining a Custom Tangent Type

The first step is to define a new type to represent the tangent of `A`. This custom tangent should “mimic” the structure of `A`, but in a way that resolves the recursion. For our example, we can define:

```@example recur_tuto
mutable struct TangentForA{Tx}
    x::Tx
    a::Union{TangentForA{Tx},Mooncake.NoTangent}

    function TangentForA{Tx}(x_tangent::Tx) where {Tx}
        new{Tx}(x_tangent, Mooncake.NoTangent())
    end

    function TangentForA{Tx}(x_tangent::Tx, a_tangent::Union{TangentForA{Tx},Mooncake.NoTangent}) where {Tx}
        new{Tx}(x_tangent, a_tangent)
    end

    function TangentForA{Tx}(nt::@NamedTuple{x::Tx, a::Union{Mooncake.NoTangent,TangentForA{Tx}}}) where Tx
    return new{Tx}(nt.x, nt.a)
end
end
```

This `TangentForA` type mirrors `A`'s fields `x` and `a`. Crucially, its `a` field is `Union{TangentForA{Tx}, Mooncake.NoTangent}`, allowing it to point to another `TangentForA` (for nested or cyclic primal structures) or terminate with `Mooncake.NoTangent` (if the primal `A.a` is `nothing`). This explicit definition of `TangentForA` as a concrete type breaks the potential infinite type recursion that would occur with a naive tangent derivation.

## Hooking into Mooncake’s Tangent Interface

Defining the tangent type is not enough—we must **register it with Mooncake’s interface** so that Mooncake knows to use it (and how to split it into fdata/rdata). We do this by implementing a few key methods:

1. **tangent\_type:** Tell Mooncake that the tangent of `A` is `TangentForA`. For example:

   ```@example recur_tuto
   function Mooncake.tangent_type(::Type{A{T}}) where {T}
        Tx = Mooncake.tangent_type(T)
        return Tx == Mooncake.NoTangent ? Mooncake.NoTangent : TangentForA{Tx}
    end
   ```

   This overrides the default mechanism and associates `A` with our custom tangent type.

2. **fdata\_type and rdata\_type:** Define the types of forward and reverse data for `TangentForA`. You need to decide which parts of `TangentForA` are treated as address-based (fdata) and which as value-based (rdata).

   In this example, because `A` is a mutable struct, its `fdata` will be the tangent itself, and `rdata` is `NoRData`. The reason is that, since the tangent type is also mutable, all updates can be done in-place.

   In some cases, you might need to carefully design the FData and RData. You can use [`Mooncake.fdata_type`](@ref) and `Mooncake.rdata_type` to tell Mooncake.

3. **tangent (combining function):** Mooncake provides a function `Mooncake.tangent(f, r)` to reassemble a tangent from fdata and rdata. For completeness, we should overload it for our types:

   ```@example recur_tuto
   Mooncake.tangent(t::TangentForA{Tx}, ::Mooncake.NoRData) where {Tx} = t
   ```

   Given our choice of `fdata` as `TangentForA` itself and `rdata` as `Mooncake.NoRData` (as is typical for mutable tangent types), the implementation `Mooncake.tangent(t::TangentForA{Tx}, ::Mooncake.NoRData) where {Tx} = t` is quite direct. The first argument `t` is the `fdata` (the `TangentForA` instance), and the second is the `rdata` (`Mooncake.NoRData`). The function simply returns the `fdata` component. This ensures that the identity `Mooncake.tangent(Mooncake.fdata(original_tangent), Mooncake.rdata(original_tangent)) === original_tangent` holds. For mutable tangents like `TangentForA`, Mooncake’s tests will check that the reassembled tangent is the exact same object as the original. This implementation guarantees such object identity, which is crucial for correctness. We will verify this with tests in Step 5.

With these interface methods, we’ve connected our custom type to Mooncake’s AD system. We have declared the tangent type and how to split/combine it.

## Bottom-Up Integration: Implement Only What You Need

Mooncake is serious about coverage of the Julia language and the quality of testing. This makes for a great user experience, but also means it might take a lot of effort to get to a point where you can pass `test_data`.

We will later show how you could approach a more complete implementation to more deeply integrate with Mooncake.

We’ll differentiate the simple function:

```@example recur_tuto
f1(a::A) = 2.0 * a.x
```

and add methods *only when Mooncake asks for them*.

### 1. Field Access (`lgetfield`) Rule

Mooncake will first complain it lacks an `rrule!!` for [`Mooncake.lgetfield`](@ref):


```@example recur_tuto
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{typeof(Mooncake.lgetfield),A{T},Val} where {T} # tell Mooncake to use the rrule!! for lgetfield, instead of trying to derive it

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake.lgetfield)},
    obj_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
    field_name_cd::Mooncake.CoDual{Val{FieldName}},
) where {T,Tx,FieldName}
    a = Mooncake.primal(obj_cd)
    a_tangent = Mooncake.tangent(obj_cd)

    value_primal = getfield(a, FieldName)

    actual_field_tangent_value = if FieldName === :x
        a_tangent.x
    elseif FieldName === :a
        a_tangent.a
    else
        throw(ArgumentError("lgetfield: Unknown field '$FieldName' for type A."))
    end

    value_output_fdata = Mooncake.fdata(actual_field_tangent_value)

    y_cd = Mooncake.CoDual(value_primal, value_output_fdata)

    function lgetfield_A_pullback(Δy_rdata)
        if FieldName === :x
            if !(Δy_rdata isa Mooncake.NoRData)
                a_tangent.x = Mooncake.increment_rdata!!(a_tangent.x, Δy_rdata)
            end
        elseif FieldName === :a
            @assert Δy_rdata isa Mooncake.NoRData # for mutable TangentForA, rdata is not used
        end

        # Return rdata for inputs: (lgetfield_func, object_a, field_name_val)
        return (Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData())
    end

    return y_cd, lgetfield_A_pullback
end
```

Next, Mooncake will want [`set_to_zero!!`](@ref):

```@example recur_tuto
function Mooncake.set_to_zero!!(t::TangentForA{Tx}) where Tx
    t.x = Mooncake.set_to_zero!!(t.x)
    if !(t.a isa Mooncake.NoTangent)
        Mooncake.set_to_zero!!(t.a)
    end
    return t
end
```

That’s enough for our first example:

```@example recur_tuto
a = A(1.0)
val, grad = DifferentiationInterface.value_and_gradient(
                f1, AutoMooncake(; config=nothing), a)

@show val           # → 2.0
@show grad.x        # → 2.0
@show grad.a        # → Mooncake.NoTangent()
```

Success!

Now it should work for functions that rely on fields of `A`.

Whenever Mooncake raises a `MethodError`, implement the missing function following the same pattern.

Depending on your use case, it may be sufficient to stop here.
```@example recur_tuto
function prod_x(a::A{T}) where {T}
    a_val = a.x
    return a.a === nothing ? a_val : a_val * prod_x(a.a)
end
sum_a = A(1.0, A(2.0, A(3.0)))
val_f5, grad_f5 = DifferentiationInterface.value_and_gradient(prod_x, AutoMooncake(; config=nothing), sum_a)
@test val_f5 == 6.0
@test grad_f5.x == 6.0
@test grad_f5.a.x == 3.0
@test grad_f5.a.a.x == 2.0
```

## From "It Works!" to Passing `test_data`

Next, we must implement fundamental operations on our tangent type so that Mooncake’s algorithms can manipulate it. At minimum, Mooncake expects the following to be defined for any custom tangent type:

Below is a checklist-style digest of **most of the functions you need to make `Mooncake.TestUtils.test_data` pass** for the recursive struct `A` and its tangent `TangentForA`.
The items are grouped by the role they play in Mooncake’s test suite.

### Primitive rrules (Mandatory Differentiation Hooks)

*Provide adjoints for every `getfield`/`lgetfield` variant that appears in tests.*

| Primitive   | Variants you implemented                                                    |
| ----------- | --------------------------------------------------------------------------- |
| `lgetfield` | `(A, Val{:x})`, `(A, Val{:a})`, plus **Symbol, Int, (Val, Val)** fallbacks. |
| `getfield`  | Same coverage as `lgetfield`.                                               |
| `_new_`   | `A(x)`, `A(x, a::A)`, `A(x, nothing)` — three separate `rrule!!`s. |
| `lsetfield!` | `(A, Val{:field}, new_value)` including both Symbol & Int field IDs. |

| Function                  | Tested feature                                                       |
| ------------------------- | -------------------------------------------------------------------- |
| `zero_tangent_internal`   | Structure-preserving zero generation with cycle cache.               |
| `randn_tangent_internal`  | Random tangent generator (for stochastic interface tests).           |
| `set_to_zero_internal!!`  | Recursive in-place reset with cycle protection.                      |
| `increment_internal!!`    | In-place accumulation used in reverse pass.                          |
| `_add_to_primal_internal` | Re-adds a tangent to a primal (needed for finite-difference checks). |
| `_diff_internal`          | Structural diff between two primals → tangent.                       |
| `_dot_internal`           | Inner-product between tangents (dual-number consistency).            |
| `_scale_internal`         | Scalar × tangent scaling.                                            |

| Override                                          | What it proves                                              |
| ------------------------------------------------- | ----------------------------------------------------------- |
| `populate_address_map_internal`                   | Tangent-to-primal pointer correspondence (cycle safety).    |
| `has_equal_data_internal` (both primal & tangent) | Deep equality ignoring pointer identity; handles recursion. |
