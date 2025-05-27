# Supporting Recursive Tangent Types in Mooncake.jl

Mooncake.jl associates each **primal type** (the original data structure) with a unique **tangent type** (the type that stores its derivative information). By default, Mooncake can automatically derive tangent types for most Julia structs. However, for *recursive types*—that is, types that reference themselves (directly or indirectly)—the default mechanism can fail, often resulting in a stack overflow. In such cases, you must manually define a custom tangent type and implement the required interface.

This guide walks you through the process, from understanding Mooncake’s tangent design to testing your custom tangent type.

## 1. Tangent Types and the FData/RData Split

Before diving in, let's review how Mooncake represents tangents (gradients) and why it splits them into **forward data** (`fdata`) and **reverse data** (`rdata`). For more details, see the [Mooncake.jl Rule System documentation](https://chalk-lab.github.io/Mooncake.jl/stable/understanding_mooncake/rule_system/).

### Tangent Types

For a given primal type `P`, `Mooncake.tangent_type(P)` returns the tangent type associated with `P`. By default, Mooncake uses generic `Tangent{...}` structs to hold fieldwise derivatives. For example, a simple struct’s tangent might be `Tangent{NamedTuple}` with the same field names as the primal. Mutable structs get a `MutableTangent` type. Each field’s tangent is itself of type `tangent_type(field_type)`.

### Forward Data vs. Reverse Data

Mooncake splits a tangent object into two parts:

- **fdata**: Forward-pass data, typically components identified by address (e.g., arrays or mutable fields), which are carried along and updated in-place.
- **rdata**: Reverse-pass data, typically value-identified components (e.g., plain numbers), only needed for the reverse pass.

This design improves performance by minimizing what needs to be propagated during the forward pass.

**Example:**  
Consider `Tuple{Float64, Vector{Float64}, Int}`. Its tangent type is `Tuple{Float64, Vector{Float64}, NoTangent}` (since `Int` is non-differentiable). The `fdata` type is `Tuple{NoFData, Vector{Float64}, NoFData}`—only the vector is forwarded. The `rdata` type is `Tuple{Float64, NoRData, NoRData}`—only the float’s derivative is carried in reverse. Mooncake ensures that for any tangent `t`, if `f = Mooncake.fdata(t)` and `r = Mooncake.rdata(t)`, then `Mooncake.tangent(f, r)` reconstructs the original `t`.

## 2. Why Recursive Types Are Challenging

A *recursive type* is a struct that contains itself (directly or indirectly) as a field. For example:

```julia
mutable struct A{T}
    x::T
    a::Union{A{T},Nothing}

    A(x::T) where {T} = new{T}(x, nothing)
    A(x::T, child::A{T}) where {T} = new{T}(x, child)
end
```

Here, `A` has a self-referential field `a`. If you ask Mooncake for the tangent type of `A{Float64}`, it tries to construct something like `Tangent{Tuple{Float64, tangent_type(A)}}`, which leads to infinite recursion. Calling `tangent_type(A)` in this scenario will overflow the stack.

To solve this, you must manually define a custom tangent type that breaks this circular dependency.

## 3. Defining a Custom Tangent Type for Recursion

The first step is to define a new type to represent the tangent of `A`. This custom tangent should mimic the structure of `A`, but in a way that resolves the recursion:

```julia
mutable struct TangentForA{Tx}
    x::Tx
    a::Union{TangentForA{Tx}, Mooncake.NoTangent}

    function TangentForA{Tx}(x_tangent::Tx) where {Tx}
        new{Tx}(x_tangent, Mooncake.NoTangent())
    end

    function TangentForA{Tx}(x_tangent::Tx, a_tangent::Union{TangentForA{Tx}, Mooncake.NoTangent}) where {Tx}
        new{Tx}(x_tangent, a_tangent)
    end

    function TangentForA{Tx}(nt::@NamedTuple{x::Tx, a::Union{Mooncake.NoTangent, TangentForA{Tx}}}) where Tx
        return new{Tx}(nt.x, nt.a)
    end
end
```

This `TangentForA` type mirrors `A`'s fields. Its `a` field is either another `TangentForA` (for nested or cyclic primal structures) or `Mooncake.NoTangent` (if the primal `A.a` is `nothing`). This explicit definition breaks the infinite type recursion that would occur with naive tangent derivation.

## 4. Registering Your Tangent Type with Mooncake

Defining the tangent type is not enough—you must **register it with Mooncake’s interface** so Mooncake knows to use it and how to split it into `fdata`/`rdata`. Implement these methods:

### 4.1. `tangent_type`

Tell Mooncake that the tangent of `A` is `TangentForA`:

```julia
function Mooncake.tangent_type(::Type{A{T}}) where {T}
    Tx = Mooncake.tangent_type(T)
    return Tx == Mooncake.NoTangent ? Mooncake.NoTangent : TangentForA{Tx}
end
```

This overrides the default mechanism and associates `A` with your custom tangent type.

### 4.2. `fdata_type` and `rdata_type`

Define the types of forward and reverse data for `TangentForA`. In this example, since both `A` and `TangentForA` are mutable, all updates can be done in-place, so the `fdata` is the tangent itself and `rdata` is `NoRData`. In other cases, you may need to split these more carefully.

### 4.3. `tangent` (Combining Function)

Mooncake provides `Mooncake.tangent(f, r)` to reassemble a tangent from `fdata` and `rdata`. For your type:

```julia
Mooncake.tangent(t::TangentForA{Tx}, ::Mooncake.NoRData) where {Tx} = t
```

This ensures that `Mooncake.tangent(Mooncake.fdata(t), Mooncake.rdata(t)) === t`, which is crucial for correctness. Mooncake’s tests will check that the reassembled tangent is the exact same object as the original.

With these methods, your custom type is now connected to Mooncake’s AD system.

## 5. Bottom-Up Integration: Implement Only What You Need

Mooncake provides extensive coverage and thorough testing. To get started, you can implement just enough to differentiate simple functions and add more as needed. For example, consider:

```julia
f1(a::A) = 2.0 * a.x
```

When you try to differentiate this, Mooncake will complain it lacks an `rrule!!` for `lgetfield`. Implement it:

### 5.1. Field Access (`lgetfield`) Rule

```julia
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{typeof(Mooncake.lgetfield),A{T},Val} where {T}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake.lgetfield)},
    obj_cd::Mooncake.CoDual{A{T},TangentForA{Tx}},
    field_name_cd::Mooncake.CoDual{Val{FieldName}},
) where {T,Tx,FieldName}
    a = Mooncake.primal(obj_cd)
    a_tangent = Mooncake.tangent(obj_cd)

    value_primal = getfield(a, FieldName)
    actual_field_tangent_value = FieldName === :x ? a_tangent.x :
                                FieldName === :a ? a_tangent.a :
                                throw(ArgumentError("lgetfield: Unknown field '$FieldName' for type A."))

    value_output_fdata = Mooncake.fdata(actual_field_tangent_value)
    y_cd = Mooncake.CoDual(value_primal, value_output_fdata)

    function lgetfield_A_pullback(Δy_rdata)
        if FieldName === :x
            if !(Δy_rdata isa Mooncake.NoRData)
                a_tangent.x = Mooncake.increment_rdata!!(a_tangent.x, Δy_rdata)
            end
        elseif FieldName === :a
            @assert Δy_rdata isa Mooncake.NoRData  # for mutable TangentForA, rdata is not used
        end
        return (Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData())
    end
    return y_cd, lgetfield_A_pullback
end
```

### 5.2. Zeroing Out the Tangent

Mooncake will next require `set_to_zero!!`:

```julia
function Mooncake.set_to_zero!!(t::TangentForA{Tx}) where Tx
    t.x = Mooncake.set_to_zero!!(t.x)
    if !(t.a isa Mooncake.NoTangent)
        Mooncake.set_to_zero!!(t.a)
    end
    return t
end
```

With these, you can now differentiate simple functions:

```julia
a = A(1.0)
val, grad = DifferentiationInterface.value_and_gradient(f1, AutoMooncake(; config=nothing), a)
@show val        # → 2.0
@show grad.x     # → 2.0
@show grad.a     # → Mooncake.NoTangent()
```

Another example:

```julia
function prod_x(a::A{T}) where {T}
    a_val = a.x
    return a.a === nothing ? a_val : a_val * prod_x(a.a)
end
sum_a = A(1.0, A(2.0, A(3.0)))
val_f5, grad_f5 = DifferentiationInterface.value_and_gradient(prod_x, AutoMooncake(; config=nothing), sum_a)
@show val_f5
@test grad_f5.x
@test grad_f5.a.x
@test grad_f5.a.a.x
```

Depending on your use case, this may be sufficient.

## 6. From "It Works!" to Passing `test_data`

To fully integrate with Mooncake, you must implement additional operations on your tangent type so Mooncake’s algorithms can manipulate it robustly. At minimum, Mooncake expects the following functions for any custom tangent type:

### Checklist: Functions Needed for Recursive Struct Support

Below is a checklist of most functions you need to make `Mooncake.TestUtils.test_data` pass for the recursive struct `A` and its tangent `TangentForA`. They are grouped by their role in Mooncake’s test suite.

#### Primitive rrules (Mandatory Differentiation Hooks)

You must provide adjoints for every `getfield`/`lgetfield` variant that appears in tests.

| Primitive     | Variants to implement                                                                           |
|-------------- |-----------------------------------------------------------------------------------------------|
| `lgetfield`   | `(A, Val{:x})`, `(A, Val{:a})`, plus Symbol, Int, and (Val, Val) fallbacks                     |
| `getfield`    | Same coverage as `lgetfield`                                                                   |
| `_new_`       | `A(x)`, `A(x, a::A)`, `A(x, nothing)`—three separate `rrule!!` methods                        |
| `lsetfield!`  | `(A, Val{:field}, new_value)` including both Symbol & Int field IDs                            |

#### Core Tangent Operations

| Function                  | Purpose/feature tested                                                      |
|---------------------------|-----------------------------------------------------------------------------|
| `zero_tangent_internal`   | Structure-preserving zero generation with cycle cache                       |
| `randn_tangent_internal`  | Random tangent generator (for stochastic interface tests)                   |
| `set_to_zero_internal!!`  | Recursive in-place reset with cycle protection                              |
| `increment_internal!!`    | In-place accumulation used in reverse pass                                  |
| `_add_to_primal_internal` | Adds a tangent to a primal (needed for finite-difference checks)            |
| `_diff_internal`          | Structural diff between two primals → tangent                               |
| `_dot_internal`           | Inner-product between tangents (dual-number consistency)                    |
| `_scale_internal`         | Scalar × tangent scaling                                                    |

#### Test Utilities

| Override                                          | What it proves                                         |
|---------------------------------------------------|--------------------------------------------------------|
| `populate_address_map_internal`                   | Tangent-to-primal pointer correspondence (cycle safety)|
| `has_equal_data_internal` (primal & tangent)      | Deep equality ignoring pointer identity; handles recursion |

By following this process—starting with a minimal set of methods and expanding as Mooncake requests more—you can support recursive types robustly in Mooncake.jl.
