# Contexts -- these are used to govern what are considered primitives.

"""
    struct MinimalCtx end

Functions should only be primitives in this context if not making them so would cause AD to
fail. In particular, do not add primitives to this context if you are writing them for
performance only.
"""
struct MinimalCtx end

is_primitive(::MinimalCtx, ::Any) = false

"""
    @is_primitive context_type signature

Creates a method of `is_primitive` which always returns `true` for the context_type and
`signature` provided. For example
```julia
@is_primitive MinimalCtx Tuple{typeof(foo), Float64}
```
is equivalent to
```julia
is_primitive(::MinimalCtx, ::Type{<:Tuple{typeof(foo), Float64}}) = true
```

You should implemented more complicated method of `is_primitive` in the usual way.
"""
macro is_primitive(Tctx, sig)
    return esc(:(Taped.is_primitive(::$Tctx, ::Type{<:$sig}) = true))
end

@is_primitive MinimalCtx Tuple{typeof(rebind), Any}

"""
    struct DefaultCtx end

Context for all usually used AD primitives. Anything which is a primitive in a MinimalCtx is
a primitive in the DefaultCtx automatically. If you are adding a rule for the sake of
performance, it should be a primitive in the DefaultCtx, but not the MinimalCtx.
"""
struct DefaultCtx end

is_primitive(::DefaultCtx, sig) = is_primitive(MinimalCtx(), sig)
