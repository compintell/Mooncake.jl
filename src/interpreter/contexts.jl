"""
    struct MinimalCtx end

Functions should only be primitives in this context if not making them so would cause AD to
fail. In particular, do not add primitives to this context if you are writing them for
performance only -- instead, make these primitives in the DefaultCtx.
"""
struct MinimalCtx end

"""
    struct DefaultCtx end

Context for all usually used AD primitives. Anything which is a primitive in a MinimalCtx is
a primitive in the DefaultCtx automatically. If you are adding a rule for the sake of
performance, it should be a primitive in the DefaultCtx, but not the MinimalCtx.
"""
struct DefaultCtx end

"""
    is_primitive(::Type{Ctx}, sig) where {Ctx}

Returns a `Bool` specifying whether the methods specified by `sig` are considered primitives
in the context of contexts of type `Ctx`.

```julia
is_primitive(DefaultCtx, Tuple{typeof(sin), Float64})
```
will return if calling `sin(5.0)` should be treated as primitive when the context is a
`DefaultCtx`.

Observe that this information means that whether or not something is a primitive in a
particular context depends only on static information, not any run-time information that
might live in a particular instance of `Ctx`.
"""
function is_primitive(::Type{MinimalCtx}, sig::Type{<:Tuple})

    # This is a very conservative hack.
    # Suppose that `sig` is `Tuple{typeof(log), Union{Float64, Int}}`.
    # We do not want to permit this call site to be inlined away, because the signature
    # `Tuple{typeof(log), Float64}` should definitely return `true`.
    # However, because this is a small Union (as opposed to a more arbitrary abstract type)
    # the compiler may choose the inline it. Consequently, we take the _very_ conservative
    # stance of asserting that any call which involves a Union should not be inlined.
    any(P -> P isa Union, sig.parameters) && return true

    return false
end
is_primitive(::Type{DefaultCtx}, sig) = is_primitive(MinimalCtx, sig)

"""
    @is_primitive context_type signature

Creates a method of `is_primitive` which always returns `true` for the context_type and
`signature` provided. For example
```julia
@is_primitive MinimalCtx Tuple{typeof(foo), Float64}
```
is equivalent to
```julia
is_primitive(::Type{MinimalCtx}, ::Type{<:Tuple{typeof(foo), Float64}}) = true
```

You should implemented more complicated method of `is_primitive` in the usual way.
"""
macro is_primitive(Tctx, sig)
    return esc(:(Mooncake.is_primitive(::Type{$Tctx}, ::Type{<:$sig}) = true))
end
