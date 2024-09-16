 """
    to_cr_tangent(t)

Convert a Tapir tangent into a type that ChainRules.jl `rrule`s expect to see.
"""
to_cr_tangent(t::IEEEFloat) = t
to_cr_tangent(t::Array{<:IEEEFloat}) = t
to_cr_tangent(::NoTangent) = ChainRulesCore.NoTangent()

"""
    increment_and_get_rdata!(fdata, rdata, cr_tangent)

"""
increment_and_get_rdata!(::NoFData, r::T, t::T) where {T<:IEEEFloat} = r + t
function increment_and_get_rdata!(f::Array{P}, ::NoRData, t::Array{P}) where {P<:IEEEFloat}
    increment!!(f, t)
    return NoRData()
end
increment_and_get_rdata!(::Any, r, ::ChainRulesCore.NoTangent) = r
function increment_and_get_rdata!(f, r, t::ChainRulesCore.Thunk)
    return increment_and_get_rdata!(f, r, ChainRulesCore.unthunk(t))
end

@doc"""
    rrule_wrapper(f::CoDual, args::CoDual...)

Used to implement `rrule!!`s via `ChainRulesCore.rrule`.

Given a function `foo`, argument types `arg_types`, and a method `ChainRulesCore.rrule` of
which applies to these, you can make use of this function as follows:
```julia
Tapir.@is_primitive DefaultCtx Tuple{typeof(foo), arg_types...}
function Tapir.rrule!!(f::CoDual{typeof(foo)}, args::CoDual...)
    return rrule_wrapper(f, args...)
end
```
Assumes that methods of `to_cr_tangent` and `to_tapir_tangent` are defined such that you
can convert between the different representations of tangents that Tapir and ChainRulesCore
expect.

Furthermore, it is _essential_ that
1. `f(args)` does not mutate `f` or `args`, and
2. the result of `f(args)` does not alias any data stored in `f` or `args`.

Subject to some constraints, you can use the [`@from_rrule`](@ref) macro to reduce the
amount of boilerplate code that you are required to write even further.
"""
function rrule_wrapper(fargs::Vararg{CoDual, N}) where {N}

    # Run forwards-pass.
    primals = tuple_map(primal, fargs)
    lazy_rdata = tuple_map(Tapir.lazy_zero_rdata, primals)
    y_primal, cr_pb = ChainRulesCore.rrule(primals...)
    y_fdata = fdata(zero_tangent(y_primal))

    function pb!!(y_rdata)

        # Construct tangent w.r.t. output.
        cr_tangent = to_cr_tangent(tangent(y_fdata, y_rdata))

        # Run reverse-pass using ChainRules.
        cr_dfargs = cr_pb(cr_tangent)

        # Increment fdata and get rdata.
        return map(fargs, lazy_rdata, cr_dfargs) do x, l_rdata, cr_dx
            return increment_and_get_rdata!(tangent(x), instantiate(l_rdata), cr_dx)
        end
    end
    return CoDual(y_primal, y_fdata), pb!!
end

function rrule_wrapper(::CoDual{typeof(Core.kwcall)}, fargs::Vararg{CoDual, N}) where {N}

    # Run forwards-pass.
    primals = tuple_map(primal, fargs)
    lazy_rdata = tuple_map(lazy_zero_rdata, primals)
    y_primal, cr_pb = Core.kwcall(primals[1], ChainRulesCore.rrule, primals[2:end]...)
    y_fdata = fdata(zero_tangent(y_primal))

    function pb!!(y_rdata)

        # Construct tangent w.r.t. output.
        cr_tangent = to_cr_tangent(tangent(y_fdata, y_rdata))

        # Run reverse-pass using ChainRules.
        cr_dfargs = cr_pb(cr_tangent)

        # Increment fdata and compute rdata.
        kwargs_rdata = rdata(zero_tangent(fargs[1]))
        args_rdata = map(fargs[2:end], lazy_rdata[2:end], cr_dfargs) do x, l_rdata, cr_dx
            return increment_and_get_rdata!(tangent(x), instantiate(l_rdata), cr_dx)
        end
        return NoRData(), kwargs_rdata, args_rdata...
    end
    return CoDual(y_primal, y_fdata), pb!!
end

@doc"""
    @from_rrule ctx sig

Convenience functionality to assist in using `ChainRulesCore.rrule`s to write `rrule!!`s.
This macro is a thin wrapper around [`rrule_wrapper`](@ref).

For example,
```julia
@from_rrule DefaultCtx Tuple{typeof(sin), Float64}
```
would define a `Tapir.rrule!!` for `sin` of `Float64`s, by calling `ChainRulesCore.rrule`.

Limitations: it is your responsibility to ensure that
1. calls with signature `sig` do not mutate their arguments,
2. the output of calls with signature `sig` does not alias any of the inputs,
3. `sig` is a `Tuple{...}`, not a `Tuple{...} where {...}`.

This last point is a limitation of the current implementation, rather than something
fundamental, whereas the first two points are more basic points.

As with all hand-written rules, you should definitely make use of
[`TestUtils.test_rule`](@ref) to verify correctness on some test cases.

# A Note On Type Constraints

Many methods of `ChainRuleCore.rrule` are implemented with very loose type constraints.
For example, it would not be surprising to see a method of rrule with the signature
```julia
Tuple{typeof(rrule), typeof(foo), Real, AbstractVector{<:Real}}
```
There are a variety of reasons for this way of doing things, and whether it is a good idea
to write rules for such generic objects has been debated at length.

Suffice it to say, you should not write rules for this package which are so generically
typed.
Rather, you should create rules for the subset of types for which you believe that the
`ChainRulesCore.rrule` will work correctly, and leave this package to derive rules for the
rest.
For example, in the above case you might be confident that the rule will behave correctly
for input types `Tuple{typeof(foo), IEEEFloat, Vector{<:IEEEFloat}}`. You should therefore
only write a rule for these types:
```julia
@from_rrule DefaultCtx Tuple{typeof(foo), IEEEFloat, Vector{<:IEEEFloat}}
```
"""
macro from_rrule(ctx, sig)

    @assert sig.head == :curly
    @assert sig.args[1] == :Tuple
    arg_type_symbols = sig.args[2:end]

    arg_names = map(n -> Symbol("x_$n"), eachindex(arg_type_symbols))
    arg_types = map(t -> :(Tapir.CoDual{<:$t}), arg_type_symbols)
    arg_exprs = map((n, t) -> :($n::$t), arg_names, arg_types)

    rule_expr = ExprTools.combinedef(
        Dict(
            :head => :function,
            :name => :(Tapir.rrule!!),
            :args => arg_exprs,
            :body => Expr(:call, rrule_wrapper, arg_names...),
        )
    )

    ex = quote
        Tapir.is_primitive(::Type{$ctx}, ::Type{<:$sig}) = true
        $rule_expr
    end
    return esc(ex)
end
