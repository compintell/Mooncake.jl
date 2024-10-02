"""
    @mooncake_overlay method_expr

Define a method of a function which only Mooncake can see. This can be used to write
versions of methods which can be successfully differentiated by Mooncake if the original
cannot be.

For example, suppose that you have a function
```jldoctest overlay
julia> foo(x::Float64) = bar(x)
foo (generic function with 1 method)
```
where Mooncake.jl fails to differentiate `bar` for some reason.
If you have access to another function `baz`, which does the same thing as `bar`, but does
    so in a way which Mooncake.jl can differentiate, you can simply write:
```jldoctest overlay
julia> Mooncake.@mooncake_overlay foo(x::Float64) = baz(x)

```
When looking up the code for `foo(::Float64)`, Mooncake.jl will see this method, rather than
the original, and differentiate it instead.

# A Worked Example

To demonstrate how to use `@mooncake_overlay`s in practice, we here demonstrate how the
answer that Mooncake.jl gives changes if you change the definition of a function using a
`@mooncake_overlay`.
Do not do this in practice -- this is just a simple way to demonostrate how to use overlays!

First, consider a simple example:
```jldoctest overlay-doctest
julia> scale(x) = 2x
scale (generic function with 1 method)

julia> rule = Mooncake.build_rrule(Tuple{typeof(scale), Float64});

julia> Mooncake.value_and_gradient!!(rule, scale, 5.0)
(10.0, (NoTangent(), 2.0))
```

We can use `@mooncake_overlay` to change the definition which Mooncake.jl sees:
```jldoctest overlay-doctest
julia> Mooncake.@mooncake_overlay scale(x) = 3x

julia> rule = Mooncake.build_rrule(Tuple{typeof(scale), Float64});

julia> Mooncake.value_and_gradient!!(rule, scale, 5.0)
(15.0, (NoTangent(), 3.0))
```
As can be seen from the output, the result of differentiating using Mooncake.jl has changed to reflect the overlay-ed definition of the method.

Additionally, it is possible to use the usual multi-line syntax to declare an overlay:
```jldoctest overlay-doctest
julia> Mooncake.@mooncake_overlay function scale(x)
           return 4x
       end

julia> rule = Mooncake.build_rrule(Tuple{typeof(scale), Float64});

julia> Mooncake.value_and_gradient!!(rule, scale, 5.0)
(20.0, (NoTangent(), 4.0))
```
"""
macro mooncake_overlay(method_expr)
    def = splitdef(method_expr)
    def[:name] = Expr(:overlay, :(Mooncake.mooncake_method_table), def[:name])
    return esc(combinedef(def))
end
