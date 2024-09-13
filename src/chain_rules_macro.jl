"""
    to_cr_tangent(t)

Convert a Tapir tangent into a type that ChainRules.jl `rrule`s expect to see.
Inverse of `to_tapir_tangent`.
"""
to_cr_tangent(t::IEEEFloat) = t
to_cr_tangent(t::Array{<:IEEEFloat}) = t
to_cr_tangent(::NoTangent) = ChainRulesCore.NoTangent()

"""
    to_tapir_tangent(cr_t)

Convert a ChainRules.jl tangent, `cr_t`, into the corresponding Tapir tangent.
Inverse of `to_cr_tangent`.
"""
to_tapir_tangent(t::IEEEFloat) = t
to_tapir_tangent(t::Array{<:IEEEFloat}) = t
to_tapir_tangent(::ChainRulesCore.NoTangent) = NoTangent()

@doc"""
    rrule_wrapper_implementation(f::CoDual, args::CoDual...)

Used to implement `rrule!!`s via `ChainRulesCore.rrule`.

Given a function `foo`, argument types `arg_types`, and a method `ChainRulesCore.rrule` of
which applies to these, you can make use of this function as follows:
```julia
Tapir.@is_primitive DefaultCtx Tuple{typeof(foo), arg_types...}
function Tapir.rrule!!(f::CoDual{typeof(foo)}, args::CoDual...)
    return rrule_wrapper_implementation(f, args...)
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
@inline function rrule_wrapper_implementation(fargs::Vararg{CoDual, N}) where {N}

    # Run forwards-pass.
    primals = tuple_map(primal, fargs)
    y_primal, cr_pb = ChainRulesCore.rrule(primals...)
    y_fdata = fdata(zero_tangent(y_primal))

    # Construct functions which, when applied to the tangent types returned on the
    # reverse-pass, will check that they are of the expected type. This will pick up on
    # obvious problems, but is intended to be fast / optimised away when things go well.
    # As such, you should think of this as a lightweight version of "debug_mode".
    tangent_type_assertions = tuple_map(
        x -> Base.Fix2(typeassert, tangent_type(typeof(x))), primals
    )

    function pb!!(y_rdata)

        # Construct tangent w.r.t. output.
        cr_tangent = to_cr_tangent(tangent(y_fdata, y_rdata))

        # Run reverse-pass using ChainRules.
        cr_dfargs = cr_pb(cr_tangent)

        # Convert output into tangent types appropriate for Tapir.
        dfargs_unvalidated = tuple_map(to_tapir_tangent, cr_dfargs)

        # Apply type assertions.
        dfargs = tuple_map((x, T) -> T(x), dfargs_unvalidated, tangent_type_assertions)

        # Increment the fdata.
        tuple_map((x, dx) -> increment!!(tangent(x), fdata(dx)), fargs, dfargs)

        # Return the rdata.
        return tuple_map(rdata, dfargs)
    end
    return CoDual(y_primal, y_fdata), pb!!
end

@doc"""
    @from_rrule ctx sig

Convenience functionality to assist in using `ChainRulesCore.rrule`s to write `rrule!!`s.
This macro is a thin wrapper around [`rrule_wrapper_implementation`](@ref).

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
            :body => Expr(:call, rrule_wrapper_implementation, arg_names...),
        )
    )

    ex = quote
        Tapir.is_primitive(::Type{$ctx}, ::Type{<:$sig}) = true
        $rule_expr
    end
    return esc(ex)
end
