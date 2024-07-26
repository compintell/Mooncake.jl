_to_rdata(::ChainRulesCore.NoTangent) = NoRData()
_to_rdata(dx::Float64) = dx

@doc"""
    @from_rrule ctx sig

Creates a `Tapir.rrule!!` from a `ChainRulesCore.rrule`. `ctx` is the type of the context in
which this rule should apply, and `sig` is the type-tuple which specifies which primal the
rule should apply to.

For example,
```julia
@from_rrule DefaultCtx Tuple{typeof(sin), Float64}
```
would define a `Tapir.rrule!!` for `sin` of `Float64`s, by calling `ChainRulesCore.rrule`.

Health warning:
Use this function with care. It has only been tested for `Float64` arguments and arguments
whose `tangent_type` is `NoTangent`, and it is entirely probable that it won't work for
arguments which aren't `Float64` or non-differentiable.

You should definitely make use of [`TestUtils.test_rule`](@ref) to verify that the rule created
works as intended.
"""
macro from_rrule(ctx, sig)

    @assert sig.head == :curly
    @assert sig.args[1] == :Tuple
    arg_type_symbols = sig.args[2:end]

    arg_names = map(n -> Symbol("x_$n"), eachindex(arg_type_symbols))
    arg_types = map(t -> :(Tapir.CoDual{<:$t}), arg_type_symbols)
    arg_exprs = map((n, t) -> :($n::$t), arg_names, arg_types)

    call_rrule = Expr(
        :call,
        :(Tapir.ChainRulesCore.rrule),
        map(n -> :(Tapir.primal($n)), arg_names)...,
    )

    pb_output_names = map(n -> Symbol("dx_$(n)_inc"), eachindex(arg_names))

    call_pb = Expr(:(=), Expr(:tuple, pb_output_names...), :(pb(dy)))
    incrementers = Expr(:tuple, map(b -> :(Tapir._to_rdata($b)), pb_output_names)...)

    pb = ExprTools.combinedef(Dict(
        :head => :function,
        :name => :pb!!,
        :args => [:dy],
        :body => quote
            $call_pb
            return $incrementers
        end,
    ))

    rule_expr = ExprTools.combinedef(
        Dict(
            :head => :function,
            :name => :(Tapir.rrule!!),
            :args => arg_exprs,
            :body => quote
                y, pb = $call_rrule
                $pb
                return Tapir.zero_fcodual(y), pb!!
            end,
        )
    )

    ex = quote
        Tapir.is_primitive(::Type{$ctx}, ::Type{$sig}) = true
        $rule_expr
    end
    return esc(ex)
end
