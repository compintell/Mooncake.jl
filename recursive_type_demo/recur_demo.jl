using Pkg
Pkg.add(path="../")

using Mooncake
using Mooncake.Random: AbstractRNG

mutable struct A{T}
    x::T
    child::A{T}  # may be undefined or self-referential
    A(x::T) where {T} = new{T}(x)            # child left uninitialized by default
    A(x::T, child::A{T}) where {T} = new{T}(x, child)
end

# TODO: make the type more concrete use the tangent type of A.x
mutable struct TangentForA{T}
    x::Any
    child::TangentForA{T}

    TangentForA{T}() where {T} = new{T}()
    function TangentForA{T}(x_tangent) where {T}
        t = new{T}()
        t.x = x_tangent
        t.child = t
        return t
    end
    # Inner constructor for full specification (child provided)
    function TangentForA{T}(x_tangent, child_tangent::TangentForA{T}) where {T}
        return new{T}(x_tangent, child_tangent)
    end
end

function Mooncake.tangent_type(::Type{A{T}}) where {T}
    if Mooncake.tangent_type(T) == Mooncake.NoTangent
        Mooncake.NoTangent
    else
        TangentForA{T}
    end
end

function Mooncake.zero_tangent_internal(a::A{T}, C::Mooncake.MaybeCache) where {T}
    Mooncake.tangent_type(A{T}) === Mooncake.NoTangent && return Mooncake.NoTangent()
    if haskey(C, a)
        return C[a]
    end
    t = TangentForA{T}()
    C[a] = t
    t.x = Mooncake.zero_tangent_internal(a.x, C)
    if isdefined(a, :child)
        t.child = Mooncake.zero_tangent_internal(a.child, C)
    else
        t.child = t
    end
    return t
end

function Mooncake.randn_tangent_internal(
    rng::AbstractRNG, x::A{T}, dict::Mooncake.MaybeCache
) where {T}
    Mooncake.tangent_type(A{T}) == Mooncake.NoTangent && return Mooncake.NoTangent()
    if Mooncake.haskey(dict, x)
        return dict[x]::TangentForA{T}
    end
    t = TangentForA{T}()
    dict[x] = t
    t.x = Mooncake.randn_tangent_internal(rng, x.x, dict)
    if isdefined(x, :child)
        t.child = Mooncake.randn_tangent_internal(rng, x.child, dict)
    end
    return t
end

function Mooncake._add_to_primal_internal(
    c::Mooncake.MaybeCache, p::A{T}, t::TangentForA{T}, unsafe::Bool
) where {T}
    Tt = Mooncake.tangent_type(A{T})
    Tt != typeof(t) && throw(
        ArgumentError(
            "p of type $(A{T}) has tangent_type $Tt, but t is of type $(typeof(t))"
        ),
    )
    # Handle circular reference by caching
    key = (p, t, unsafe)
    if Mooncake.haskey(c, key)
        return c[key]::A{T}
    end
    # Construct initial perturbed primal:
    # Use existing inner constructor A(x) to set x and leave child undefined
    new_x = Mooncake._add_to_primal_internal(c, p.x, t.x, unsafe)
    p_new = A{T}(new_x)
    c[key] = p_new
    # Handle child field
    if isdefined(p, :child)
        if Mooncake.is_init(t.child)  # t.child always init in our custom type if p.child defined
            p_new.child = Mooncake._add_to_primal_internal(c, p.child, t.child, unsafe)
        else
            # Both child undefined (no child in both p and t): nothing to set
        end
    elseif Mooncake.is_init(t.child)
        throw(
            error(
                "unable to handle undefined-ness: primal child missing but tangent child present",
            ),
        )
    end
    return p_new
end

### Differencing two A's (for testing correctness) ###

function Mooncake._diff_internal(c::Mooncake.MaybeCache, p::A{T}, q::A{T}) where {T}
    Mooncake.tangent_type(A{T}) === Mooncake.NoTangent && return Mooncake.NoTangent()
    if Mooncake.haskey(c, (p, q))
        return c[(p, q)]::TangentForA{T}
    end
    # Create empty tangent for result
    t = TangentForA{T}()
    c[(p, q)] = t
    # Compute field differences
    t.x = Mooncake._diff_internal(c, p.x, q.x)
    if isdefined(p, :child) && isdefined(q, :child)
        t.child = Mooncake._diff_internal(c, p.child, q.child)
    else
        # If both children undefined, leave t.child uninitialized (no difference for that field)
        !isdefined(p, :child) && !isdefined(q, :child) ||
            throw(error("Unhandleable undefinedness in _diff_internal"))
    end
    # If both fields are NoTangent, compress to NoTangent (all differences zero)
    if t.x === Mooncake.NoTangent() && (!isdefined(t, :child) || !Mooncake.is_init(t.child))
        return Mooncake.NoTangent()
    end
    return t
end

### In-place Tangent Operations ###

function Mooncake.increment_internal!!(
    c::Mooncake.IncCache, x::TangentForA{T}, y::TangentForA{T}
) where {T}
    if x === y || Mooncake.haskey(c, x)
        return x
    end
    c[x] = true
    # Add fields
    x.x = Mooncake.increment_internal!!(c, x.x, y.x)
    if isdefined(x, :child) && isdefined(y, :child)
        x.child = Mooncake.increment_internal!!(c, x.child, y.child)
    elseif !isdefined(x, :child) && !isdefined(y, :child)
        # no child in either -> nothing to do
    else
        throw(error("Mismatched child structure in increment_internal!!"))
    end
    return x
end

function Mooncake.set_to_zero_internal!!(c::Mooncake.IncCache, t::TangentForA{T}) where {T}
    if Mooncake.haskey(c, t)
        return t
    end
    c[t] = false  # mark visited
    t.x = Mooncake.set_to_zero_internal!!(c, t.x)
    if isdefined(t, :child)
        # Recurse on child (which may be self or a sub-tangent)
        Mooncake.set_to_zero_internal!!(c, t.child)
        # (No need to reassign t.child; mutated in-place if not self)
    end
    return t
end

### Inner Product and Scalar Scaling ###

function Mooncake._dot_internal(
    c::Mooncake.MaybeCache, t::TangentForA{T}, s::TangentForA{T}
) where {T}
    key = (t, s)
    if Mooncake.haskey(c, key)
        return c[key]::Float64
    end
    c[key] = 0.0
    # Compute dot: t.x·s.x + t.child·s.child
    acc = Mooncake._dot_internal(c, t.x, s.x)
    if isdefined(t, :child) && isdefined(s, :child)
        acc += Mooncake._dot_internal(c, t.child, s.child)
    end
    return acc
end

function Mooncake._scale_internal(
    c::Mooncake.MaybeCache, a::Float64, t::TangentForA{T}
) where {T}
    if Mooncake.haskey(c, t)
        return c[t]::TangentForA{T}
    end
    # Create new tangent for result
    u = TangentForA{T}()
    c[t] = u
    u.x = Mooncake._scale_internal(c, a, t.x)
    if isdefined(t, :child)
        u.child = Mooncake._scale_internal(c, a, t.child)
    end
    return u
end

Mooncake.tangent(f::TangentForA{T}, ::Mooncake.NoRData) where {T} = f

### Custom lgetfield rule for A ###

# Define how TangentForA interacts with Mooncake's fdata/rdata system
# Assumes TangentForA acts like a MutableTangent: its fdata is itself, its rdata is NoRData.
Mooncake.fdata_type(::Type{TangentForA{T}}) where {T} = TangentForA{T}
Mooncake.rdata_type(::Type{TangentForA{T}}) where {T} = Mooncake.NoRData

# Mark lgetfield on A as a primitive operation for Mooncake's AD
# This tells Mooncake to use our rrule!! and not try to differentiate lgetfield further for A.
# Note: Adjust context (e.g., DefaultCtx) if you use a different one.
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(Mooncake.lgetfield),A,Val}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake.lgetfield)},
    obj_cd::Mooncake.CoDual{A{T},TangentForA{T}}, # obj_cd.tangent is TangentForA
    field_name_cd::Mooncake.CoDual{Val{FieldName}},
) where {T,FieldName}
    a = Mooncake.primal(obj_cd)
    a_tangent = Mooncake.tangent(obj_cd) # This is an instance of TangentForA{T}

    # --- Forward Pass ---
    value_primal = getfield(a, FieldName)

    # Determine the tangent of the field being accessed
    actual_field_tangent_value = if FieldName === :x
        a_tangent.x
    elseif FieldName === :child
        a_tangent.child # This is another TangentForA instance
    else
        # This case should ideally be prevented by Julia's dispatch if FieldName is concrete
        error("lgetfield: Unknown field '$FieldName' for type A.")
    end

    # Determine the fdata for the *output value* of the getfield operation.
    # fdata(tangent_of_the_output_value)
    value_output_fdata = Mooncake.fdata(actual_field_tangent_value)

    y_cd = Mooncake.CoDual(value_primal, value_output_fdata)

    # --- Backward Pass (Pullback) ---
    function lgetfield_A_pullback(Δy_rdata)
        # Δy_rdata is the gradient (rdata) for the output of getfield.

        # Gradients for fields of `a` are accumulated into `a_tangent`.
        if FieldName === :x
            if !(Δy_rdata isa Mooncake.NoRData)
                # a_tangent.x is the tangent for a.x (e.g., a Float64)
                # Δy_rdata is the gradient for a.x (e.g., a Float64)
                # increment_rdata!! for Float64s is typically addition.
                a_tangent.x = Mooncake.increment_rdata!!(a_tangent.x, Δy_rdata)
            end
        elseif FieldName === :child
            # a_tangent.child is the tangent for a.child (a TangentForA instance).
            # If rdata_type(TangentForA{T}) is NoRData, then Δy_rdata will be NoRData().
            # In this case, increment_rdata!!(a_tangent.child, NoRData()) should be a no-op
            # or return a_tangent.child unchanged. The gradient is considered propagated
            # "to" a_tangent.child because it's the tangent object for a mutable structure.
            if !(Δy_rdata isa Mooncake.NoRData)
                # This would only happen if rdata_type(TangentForA) was not NoRData.
                # It would mean Δy_rdata needs to be incorporated into a_tangent.child.
                a_tangent.child = Mooncake.increment_rdata!!(a_tangent.child, Δy_rdata)
            end
        end

        # Return rdata for inputs: (lgetfield_func, object_a, field_name_val)
        # NoRData for lgetfield func and Val{FieldName}.
        # NoRData for object `a` because its tangent `a_tangent` was mutated directly.
        return (Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData())
    end

    return y_cd, lgetfield_A_pullback
end

##

# Helper to initialize a self-referential A (circular)
function self_ref_A(x::T) where {T}
    a = A(x)
    a.child = a   # make it point to itself
    return a
end

# Example differentiable functions using A
f(a::A{Float64}) = 2.0 * a.x                      # linear function of A.x
g(a::A{Float64}) = a.x^2                          # nonlinear function
h(a::A{Float64}) = a.x + (isdefined(a, :child) ? a.child.x : 0.0)  # uses child.x if present

# Prepare a self-referential instance
a = self_ref_A(3.0)

# Differentiate f, g, and h at point a using Mooncake
rule_f = Mooncake.build_rrule(Tuple{typeof(f),A{Float64}})
val_f, (_, ∂f_∂a) = Mooncake.value_and_gradient!!(rule_f, f, a)  # gradient returns a tuple: (grad_for_func, grad_for_arg1)
println("f(a) = $val_f,   grad f = $(∂f_∂a_actual) with ∂x = $(∂f_∂a_actual.x)")

rule_g = Mooncake.build_rrule(Tuple{typeof(g),A{Float64}})
val_f, (_, ∂f_∂a) = Mooncake.value_and_gradient!!(rule_g, g, a)

rule_h = Mooncake.build_rrule(Tuple{typeof(h),A{Float64}})
val_h, (_, ∂h_∂a) = Mooncake.value_and_gradient!!(rule_h, h, a)
∂h_∂a.child === ∂h_∂a
