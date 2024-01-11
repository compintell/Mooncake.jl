"""
    normalise!(ir::IRCode)

Apply a sequence of standardising transformations to `ir` which leaves its semantics
unchanged, but make AD more straightforward. In particular, replace
1. `:invoke` `Expr`s with `:call`s,
2. `:foreigncall` `Expr`s with `:call`s to `Taped._foreigncall_`,
3. `:new` `Expr`s with `:call`s to `Taped._new_`,
4. `Core.IntrinsicFunction`s with counterparts from `Taped.IntrinsicWrappers`.

After these standardisations are applied, applies `ensure_single_argument_usage_per_call!`,
which is _essential_ for the correctness of AD.
"""
function normalise!(ir::IRCode)
    invokes_to_calls!(ir)
    foreigncall_exprs_to_call_exprs!(ir)
    new_exprs_to_call_exprs!(ir)
    intrinsic_calls_to_function_calls!(ir)
    ensure_single_argument_usage_per_call!(ir)
    return ir
end

"""
    invokes_to_calls!(ir::IRCode)

Replaces all instances of `:invoke` expressions with equivalent `:call` expressions.
"""
function invokes_to_calls!(ir::IRCode)
    for inst in ir.stmts.inst
        if Meta.isexpr(inst, :invoke)

            # A _sufficient_ condition for it to be safe to perform this transformation is
            # that the types in the `MethodInstance` that this `:invoke` node refers to are
            # concrete. Check this. It _might_ be that there exist `:invoke` nodes which do
            # not satisfy this criterion, but for which it is safe to perform this
            # transformation -- if this is true, I (Will) have yet to encounter one.
            mi = inst.args[1]
            @assert all(isconcretetype, mi.specTypes.parameters)

            # Change to a call.
            inst.head = :call
            inst.args = inst.args[2:end]
        end
    end
    return ir
end

"""
    foreigncall_exprs_to_call_exprs!(ir::IRCode)

Replace all instance of `:foreigncall` expressions with equivalent `:call` expressions.
"""
function foreigncall_exprs_to_call_exprs!(ir::IRCode)
    foreach(__foreigncall_expr_to_call_expr!, ir.stmts.inst)
end

# If `inst` is a :foreigncall expression, translate it into an equivalent :call expression.
function __foreigncall_expr_to_call_expr!(inst)
    if Meta.isexpr(inst, :foreigncall)
        args = inst.args
        name = __extract_foreigncall_name(args[1])
        sparams_dict = Dict(zip(spnames, sptypes))
        RT = Val(interpolate_sparams(args[2], sparams_dict))
        AT = (map(x -> Val(interpolate_sparams(x, sparams_dict)), args[3])..., )
        nreq = Val(args[4])
        calling_convention = Val(args[5] isa QuoteNode ? args[5].value : args[5])
        x = args[6:end]
        inst.head = :call
        f = GlobalRef(Taped, :_foreigncall_)
        inst.args = Any[f, name, RT, AT, nreq, calling_convention, x...]
    end
    return inst
end

# Copied from Umlaut.jl.
__extract_foreigncall_name(x::Symbol) = Val(x)
function __extract_foreigncall_name(x::Expr)
    # Make sure that we're getting the expression that we're expecting.
    !Meta.isexpr(x, :call) && error("unexpected expr $x")
    !isa(x.args[1], GlobalRef) && error("unexpected expr $x")
    x.args[1].name != :tuple && error("unexpected expr $x")
    length(x.args) != 3 && error("unexpected expr $x")

    # Parse it into a name that can be passed as a type.
    v = eval(x)
    return Val((Symbol(v[1]), Symbol(v[2])))
end
__extract_foreigncall_name(v::Tuple) = Val((Symbol(v[1]), Symbol(v[2])))
__extract_foreigncall_name(x::QuoteNode) = __extract_foreigncall_name(x.value)
function __extract_foreigncall_name(x::GlobalRef)
    return __extract_foreigncall_name(getglobal(x.mod, x.name))
end

# Copied from Umlaut.jl. Originally, adapted from
# https://github.com/JuliaDebug/JuliaInterpreter.jl/blob/aefaa300746b95b75f99d944a61a07a8cb145ef3/src/optimize.jl#L239
function interpolate_sparams(@nospecialize(t::Type), sparams::Dict)
    t isa Core.TypeofBottom && return t
    while t isa UnionAll
        t = t.body
    end
    t = t::DataType
    if Base.isvarargtype(t)
        return Expr(:(...), t.parameters[1])
    end
    if Base.has_free_typevars(t)
        params = map(t.parameters) do @nospecialize(p)
            if isa(p, TypeVar)
                return sparams[p.name].typ.val
            elseif isa(p, DataType) && Base.has_free_typevars(p)
                return interpolate_sparams(p, sparams)
            elseif p isa CC.VarState
                @show "doing varstate"
                p.typ
            else
                return p
            end
        end
        T = t.name.Typeofwrapper.parameters[1]
        return T{params...}
    end
    return t
end

"""
    new_exprs_to_call_exprs!(ir::IRCode)

Replaces all `:new` expressions with `:call` expressions to `Taped._new_`.
"""
function new_exprs_to_call_exprs!(ir::IRCode)
    foreach(__new_expr_to_call_expr!, ir.stmts.inst)
end

function __new_expr_to_call_expr!(ex::Expr)
    if Meta.isexpr(ex, :new)
        ex.head = :call
        ex.args = [_new_, ex.args...]
    end
end

function _lift_expr_arg(ex::Expr, sptypes)
    if Meta.isexpr(ex, :boundscheck)
        return ConstSlot(true)
    elseif Meta.isexpr(ex, :static_parameter)
        out_type = sptypes[ex.args[1]]
        if out_type isa CC.VarState
            out_type = out_type.typ
        end
        return ConstSlot(out_type)
    else
        throw(ArgumentError("Found unexpected expr $ex"))
    end
end

# _lift_intrinsic(x) = x
# function _lift_intrinsic(x::Core.IntrinsicFunction)
#     return x == cglobal ? x : IntrinsicsWrappers.translate(Val(x))
# end

# _lift_expr_arg(ex::Union{Argument, SSAValue, CC.MethodInstance}, _) = ex
# _lift_expr_arg(ex::QuoteNode, _) = ConstSlot(_lift_intrinsic(ex.value))
# _lift_expr_arg(ex::GlobalRef, _) = _globalref_to_slot(ex)

# function _globalref_to_slot(ex::GlobalRef)
#     val = getglobal(ex.mod, ex.name)
#     val isa Core.IntrinsicFunction && return ConstSlot(_lift_intrinsic(val))
#     isconst(ex) && return ConstSlot(val)
#     return TypedGlobalRef(ex)
# end

# _lift_expr_arg(ex, _) = ConstSlot(ex)

# _lift_expr_arg(ex::AbstractSlot, _) = throw(ArgumentError("ex is already a slot!"))

"""
    intrinsic_calls_to_function_calls!(ir::IRCode)

Replace `:call`s to `Core.IntrinsicFunction`s with `:call`s to counterpart functions in
`Taped.IntrinsicWrappers`. These wrappers are primitives, and have `rrule!!`s written for
them directly.
"""
function intrinsic_calls_to_function_calls!(ir::IRCode)
    sptypes = ir.sptypes
    spnames = in_f.spnames # need method for this stuff
    for inst in ir.stmts.inst
        if Meta.isexpr(inst, :call)
            ex.args = map(Base.Fix2(_lift_expr_arg, sptypes), ex.args)
        end
    end
    return ir
end

"""
    ensure_single_argument_usage_per_call!(ir::IRCode)

Transforms `ir` to ensures that all `:call` expressions have only a single usage of any
`Argument` or `SSAValue`.

For example, an expression such as
```julia
foo(%1, %2, %1)
```
uses the `SSAValue` `%1` twice. This can cause correctness issues issues on the reverse-pass
of AD if `%1` happens to be differentiable and a bits-type.

`ensure_single_argument_usage_per_call!` transforms the above example, and generalisations
thereof, into
```julia
x = __rebind(%1)
foo(%1, %2, x)
```
where `__rebind` is equivalent to the `identity`, but is always a primitive.
"""
function ensure_single_argument_usage_per_call!(ir::IRCode)

end

__rebind(x) = x
@is_primitive MinimalCtx Tuple{typeof(__rebind), Any}
__rebind_pb!!(dy, df, dx) = df, increment!!(dx, dy)
rrule!!(::CoDual{typeof(__rebind)}, x::CoDual) = x, __rebind_pb!!
