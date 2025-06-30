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
    abstract type Mode end

Subtypes of this signify which mode of AD is being considered.
"""
abstract type Mode end

"""
    struct ForwardMode end

Used primarily as the second argument to [`is_primitive`](@ref) to determine whether a
function is a primitive in forwards-mode AD.
"""
struct ForwardMode <: Mode end

"""
    struct ReverseMode end

Used primarily as the second argument to [`is_primitive`](@ref) to determine whether a
function is a primitive in reverse-mode AD.
"""
struct ReverseMode <: Mode end

"""
    is_primitive(::Type{Ctx}, ::Type{Mode}, sig) where {Ctx,Mode}

Returns a `Bool` specifying whether the methods specified by `sig` are considered primitives
in the context of contexts of type `Ctx` in `Mode`.

```julia
is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(sin), Float64})
```
will return if calling `sin(5.0)` should be treated as primitive when the context is a
`DefaultCtx`.

Observe that this information means that whether or not something is a primitive in a
particular context depends only on static information, not any run-time information that
might live in a particular instance of `Ctx`.
"""
is_primitive(::Type{MinimalCtx}, ::Type{<:Mode}, sig::Type{<:Tuple}) = false
is_primitive(::Type{DefaultCtx}, M::Type{<:Mode}, sig) = is_primitive(MinimalCtx, M, sig)

"""
    @is_primitive context_type signature

Creates a method of `is_primitive` which always returns `true` for the `context_type`, and
`signature` provided. For example
```julia
@is_primitive MinimalCtx Tuple{typeof(foo), Int}
```
is equivalent to
```julia
is_primitive(::Type{MinimalCtx}, ::Type{<:Mode}, ::Type{<:Tuple{typeof(foo), Int}}) = true
```
Observe that this means that a rule is a primitive in all AD modes. See the three-argument
version of this macro for more control over the mode specified.

You should implemented more complicated method of `is_primitive` in the usual way.
"""
macro is_primitive(Tctx, sig)
    return _is_primitive_expression(Tctx, :(Mooncake.Mode), sig)
end

macro is_primitive(Tctx, Tmode, sig)
    return _is_primitive_expression(Tctx, esc(Tmode), sig)
end

function _is_primitive_expression(Tctx, Tmode, sig)
    return quote
        function Mooncake.is_primitive(
            ::Type{$(esc(Tctx))}, ::Type{<:$(Tmode)}, ::Type{<:$(esc(sig))}
        )
            return true
        end
    end
end
