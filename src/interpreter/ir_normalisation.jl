"""
    normalise!(ir::IRCode)

Apply a sequence of standardising transformations to `ir` which leaves its semantics
unchanged, but make AD more straightforward. In particular, replace
1. `:invoke` `Expr`s with `:call`s,
2. `:foreigncall` `Expr`s with `:call`s to `Taped._foreigncall_`,
3. `:new` `Expr`s with `:call`s to `Taped._new_`,
4. `Core.IntrinsicFunction`s with counterparts from `Taped.IntrinsicWrappers`.

After these standardisations are applied, applies `ensure_single_argument_usage_per_call!`,
which is essential for the correctness of AD.
"""
function normalise!(ir::IRCode)


    return ensure_single_argument_usage_per_call!(ir)
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
x = __single_use_shim(%1)
foo(%1, %2, x)
```
where `__single_use_shim` is equivalent to the `identity`, but is always a primitive.
"""
function ensure_single_argument_usage_per_call!(ir::IRCode)

end

# Replaces all instances of `:invoke` expressions with equivalent `:call` expressions.
function __invokes_to_calls!(ir::IRCode)
    insts::Vector{Any} = ir.stmts.inst
    for (n, inst) in enumerate(insts)
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

# function preprocess_ir(ex::Expr, in_f)
#     ex = CC.copy(ex)
#     sptypes = in_f.ir.sptypes
#     spnames = in_f.spnames
#     if Meta.isexpr(ex, :foreigncall)
#         args = ex.args
#         name = extract_foreigncall_name(args[1])
#         sparams_dict = Dict(zip(spnames, sptypes))
#         RT = Val(interpolate_sparams(args[2], sparams_dict))
#         AT = (map(x -> Val(interpolate_sparams(x, sparams_dict)), args[3])..., )
#         nreq = Val(args[4])
#         calling_convention = Val(args[5] isa QuoteNode ? args[5].value : args[5])
#         x = args[6:end]
#         ex.head = :call
#         f = GlobalRef(Taped, :_foreigncall_)
#         ex.args = Any[f, name, RT, AT, nreq, calling_convention, x...]
#         ex.args = map(Base.Fix2(_lift_expr_arg, sptypes), ex.args)
#         return ex
#     elseif Meta.isexpr(ex, :new)
#         ex.head = :call
#         ex.args = map(Base.Fix2(_lift_expr_arg, sptypes), [_new_, ex.args...])
#         return ex
#     else
#         ex.args = map(Base.Fix2(_lift_expr_arg, sptypes), ex.args)
#         return ex
#     end
# end
