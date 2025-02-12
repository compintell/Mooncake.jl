function build_frule(args...; debug_mode=false)
    interp = get_interpreter()
    sig = _typeof(TestUtils.__get_primals(args))
    return build_frule(interp, sig; debug_mode)
end

function build_frule(
    interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode=false, silence_debug_messages=true
) where {C}
    # To avoid segfaults, ensure that we bail out if the interpreter's world age is greater
    # than the current world age.
    if Base.get_world_counter() > interp.world
        throw(
            ArgumentError(
                "World age associated to interp is behind current world age. Please " *
                "a new interpreter for the current world age.",
            ),
        )
    end

    # If we're compiling in debug mode, let the user know by default.
    if !silence_debug_messages && debug_mode
        @info "Compiling rule for $sig_or_mi in debug mode. Disable for best performance."
    end

    # If we have a hand-coded rule, just use that.
    sig = _get_sig(sig_or_mi)
    is_primitive(C, sig) && return (debug_mode ? DebugFRule(frule!!) : frule!!)

    # We don't have a hand-coded rule, so derived one.
    lock(MOONCAKE_INFERENCE_LOCK)
    try
        # If we've already derived the OpaqueClosures and info, do not re-derive, just
        # create a copy and pass in new shared data.
        oc_cache_key = ClosureCacheKey(interp.world, (sig_or_mi, debug_mode))
        # if haskey(interp.oc_cache, oc_cache_key)
        #     return interp.oc_cache[oc_cache_key]
        # else
        # Derive forward-pass IR, and shove in a `MistyClosure`.
        dual_ir = generate_dual_ir(interp, sig_or_mi; debug_mode)
        dual_oc = MistyClosure(dual_ir; do_compile=true)
        raw_rule = DerivedFRule(dual_oc)
        rule = debug_mode ? DebugFRule(raw_rule) : raw_rule
        interp.oc_cache[oc_cache_key] = rule
        return rule
        # end
    catch e
        rethrow(e)
    finally
        unlock(MOONCAKE_INFERENCE_LOCK)
    end
end

struct DerivedFRule{Tfwd_oc}
    fwd_oc::Tfwd_oc
end

@inline function (fwd::DerivedFRule)(args::Vararg{Dual,N}) where {N}
    return fwd.fwd_oc(args...)
end

function generate_dual_ir(
    interp::MooncakeInterpreter, sig_or_mi; debug_mode=false, do_inline=true
)
    # Reset id count. This ensures that the IDs generated are the same each time this
    # function runs.
    seed_id!()

    # Grab code associated to the primal.
    primal_ir, _ = lookup_ir(interp, sig_or_mi)

    # Normalise the IR.
    _, spnames = is_vararg_and_sparam_names(sig_or_mi)
    primal_ir = normalise!(primal_ir, spnames)

    # Keep a copy of the primal IR with the insertions
    dual_ir = copy(primal_ir)

    # Modify dual argument types:
    # - add one for the rule in front
    # - convert the rest to dual types
    for (a, P) in enumerate(primal_ir.argtypes)
        if P isa DataType
            dual_ir.argtypes[a] = dual_type(P)
        elseif P isa Core.Const
            dual_ir.argtypes[a] = dual_type(_typeof(P.val))
        end
    end
    pushfirst!(dual_ir.argtypes, Any)

    # Modify dual IR incrementally (do the same over primal because otherwise they won't match due to nothing statements)
    primal_ir_comp = CC.IncrementalCompact(primal_ir)
    dual_ir_comp = CC.IncrementalCompact(dual_ir)

    for (((_, primal_ssa), primal_stmt), ((_, dual_ssa), dual_stmt)) in
        zip(primal_ir_comp, dual_ir_comp)
        modify_fwd_ad_stmts!(
            dual_stmt,
            primal_stmt,
            dual_ir_comp,
            primal_ir_comp,
            dual_ssa,
            primal_ssa,
            interp;
            debug_mode,
        )
    end
    dual_ir_comp = CC.finish(dual_ir_comp)
    dual_ir_comp = CC.compact!(dual_ir_comp)

    # display(primal_ir)

    CC.verify_ir(dual_ir_comp)

    # Optimize dual IR
    dual_ir_opt = optimise_ir!(dual_ir_comp; do_inline)  # TODO: toggle
    return dual_ir_opt
end

## Modification of IR nodes

const REVERSE_AFFINITY = true  # stick the new instruction in the previous CFG block (incremental insertion cannot be done before "where we are")

function MyInstruction(stmt)
    return CC.NewInstruction(
        stmt,
        Any,
        CC.NoCallInfo(),
        Int32(1),  # meaningless
        CC.IR_FLAG_REFINED,
    )
end

function modify_fwd_ad_stmts!(
    dual_stmt::Nothing,
    primal_stmt::Nothing,
    dual_ir::CC.IncrementalCompact,
    primal_ir::CC.IncrementalCompact,
    dual_ssa::Integer,
    primal_ssa::Integer,
    ::MooncakeInterpreter;
    kwargs...,
)
    return nothing
end

function modify_fwd_ad_stmts!(
    dual_stmt::GotoNode,
    primal_stmt::GotoNode,
    dual_ir::CC.IncrementalCompact,
    primal_ir::CC.IncrementalCompact,
    dual_ssa::Integer,
    primal_ssa::Integer,
    ::MooncakeInterpreter;
    kwargs...,
)
    return nothing
end

function modify_fwd_ad_stmts!(
    dual_stmt::GotoIfNot,
    primal_stmt::GotoIfNot,
    dual_ir::CC.IncrementalCompact,
    primal_ir::CC.IncrementalCompact,
    dual_ssa::Integer,
    primal_ssa::Integer,
    ::MooncakeInterpreter;
    kwargs...,
)
    # replace GotoIfNot with the call to primal
    Mooncake.replace_call!(
        dual_ir, SSAValue(dual_ssa), Expr(:call, _primal, inc_args(dual_stmt).cond)
    )
    # reinsert the GotoIfNot right after the call to primal
    new_gotoifnot_inst = MyInstruction(Core.GotoIfNot(SSAValue(dual_ssa), dual_stmt.dest))
    CC.insert_node_here!(dual_ir, new_gotoifnot_inst, REVERSE_AFFINITY)
    return nothing
end

function modify_fwd_ad_stmts!(
    dual_stmt::GlobalRef,
    primal_stmt::GlobalRef,
    dual_ir::CC.IncrementalCompact,
    primal_ir::CC.IncrementalCompact,
    dual_ssa::Integer,
    primal_ssa::Integer,
    ::MooncakeInterpreter;
    kwargs...,
)
    return nothing
end

function modify_fwd_ad_stmts!(
    dual_stmt::ReturnNode,
    primal_stmt::ReturnNode,
    dual_ir::CC.IncrementalCompact,
    primal_ir::CC.IncrementalCompact,
    dual_ssa::Integer,
    primal_ssa::Integer,
    ::MooncakeInterpreter;
    kwargs...,
)
    # make sure that we always return a Dual even when it's a constant
    if isdefined(primal_stmt, :val)
        Mooncake.replace_call!(
            dual_ir, SSAValue(dual_ssa), Expr(:call, _dual, inc_args(dual_stmt).val)
        )
    else
        # a ReturnNode without a val is an unreachable
        nothing
    end
    # return the result from the previous Dual conversion
    new_return_inst = MyInstruction(ReturnNode(SSAValue(dual_ssa)))
    CC.insert_node_here!(dual_ir, new_return_inst, REVERSE_AFFINITY)
    return nothing
end

function modify_fwd_ad_stmts!(
    dual_stmt::PhiNode,
    primal_stmt::PhiNode,
    dual_ir::CC.IncrementalCompact,
    primal_ir::CC.IncrementalCompact,
    dual_ssa::Integer,
    primal_ssa::Integer,
    ::MooncakeInterpreter;
    kwargs...,
)
    dual_ir[SSAValue(dual_ssa)][:stmt] = inc_args(dual_stmt)
    # TODO: translate constants like GlobalRef into constant Duals
    dual_ir[SSAValue(dual_ssa)][:type] = Any
    dual_ir[SSAValue(dual_ssa)][:flag] = CC.IR_FLAG_REFINED
    return nothing
end

function modify_fwd_ad_stmts!(
    dual_stmt::PiNode,
    primal_stmt::PiNode,
    dual_ir::CC.IncrementalCompact,
    primal_ir::CC.IncrementalCompact,
    dual_ssa::Integer,
    primal_ssa::Integer,
    ::MooncakeInterpreter;
    kwargs...,
)
    dual_ir[SSAValue(dual_ssa)][:stmt] = PiNode(inc_args(dual_stmt).val, Any)  # TODO: deduce proper dual type (impossible now because PhiNodes may return undualized values with GlobalRefs)
    dual_ir[SSAValue(dual_ssa)][:type] = Any
    dual_ir[SSAValue(dual_ssa)][:flag] = CC.IR_FLAG_REFINED
    return nothing
end

## Modification of IR nodes - expressions

struct DualArguments{FR}
    frule::FR
end

function Base.show(io::IO, da::DualArguments)
    return print(io, "DualArguments($(da.frule))")
end

# TODO: wrapping in Dual must not be systematic (e.g. Argument or SSAValue)
function (da::DualArguments)(f::F, args::Vararg{Any,N}) where {F,N}
    return da.frule(tuple_map(_dual, (f, args...))...)
end

function modify_fwd_ad_stmts!(
    dual_stmt::Expr,
    primal_stmt::Expr,
    dual_ir::CC.IncrementalCompact,
    primal_ir::CC.IncrementalCompact,
    dual_ssa::Integer,
    primal_ssa::Integer,
    interp::MooncakeInterpreter;
    debug_mode,
)
    stmt = dual_stmt

    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        sig, mi = if isexpr(stmt, :invoke)
            mi = stmt.args[1]::Core.MethodInstance
            mi.specTypes, mi
        else
            sig_types = map(primal_stmt.args) do primal_arg
                T = get_forward_primal_type(primal_ir, primal_arg)
                return T
            end
            Tuple{sig_types...}, missing
        end
        shifted_args = if isexpr(stmt, :invoke)
            inc_args(stmt).args[2:end]  # first arg is method instance
        else
            inc_args(stmt).args
        end
        if is_primitive(context_type(interp), sig)
            call_frule = Expr(:call, DualArguments(frule!!), shifted_args...)
            replace_call!(dual_ir, SSAValue(dual_ssa), call_frule)
        else
            if isexpr(stmt, :invoke)
                rule = LazyFRule(mi, debug_mode)
            else
                rule = DynamicFRule(debug_mode)
            end
            # TODO: could this insertion of a naked rule in the IR cause a memory leak?
            call_rule = Expr(:call, DualArguments(rule), shifted_args...)
            replace_call!(dual_ir, SSAValue(dual_ssa), call_rule)
        end
    elseif isexpr(stmt, :boundscheck) || isexpr(stmt, :loopinfo)
        nothing
    elseif isexpr(stmt, :code_coverage_effect)
        replace_call!(dual_ir, SSAValue(dual_ssa), nothing)
    else
        throw(
            ArgumentError(
                "Expressions of type `:$(stmt.head)` are not yet supported in forward mode"
            ),
        )
    end
    return nothing
end

function get_forward_primal_type(ir::CC.IncrementalCompact, a::Argument)
    return ir.ir.argtypes[a.n]
end

function get_forward_primal_type(ir::CC.IncrementalCompact, ssa::SSAValue)
    return ir[ssa][:type]
end

function get_forward_primal_type(::CC.IncrementalCompact, x::QuoteNode)
    return _typeof(x.value)
end

function get_forward_primal_type(::CC.IncrementalCompact, x)
    return _typeof(x)
end

function get_forward_primal_type(::CC.IncrementalCompact, x::GlobalRef)
    return isconst(x) ? _typeof(getglobal(x.mod, x.name)) : x.binding.ty
end

function get_forward_primal_type(::CC.IncrementalCompact, x::Expr)
    x.head === :boundscheck && return Bool
    return error("Unrecognised expression $x found in argument slot.")
end

mutable struct LazyFRule{primal_sig,Trule}
    debug_mode::Bool
    mi::Core.MethodInstance
    rule::Trule
    function LazyFRule(mi::Core.MethodInstance, debug_mode::Bool)
        interp = get_interpreter()
        return new{mi.specTypes,frule_type(interp, mi; debug_mode)}(debug_mode, mi)
    end
    function LazyFRule{Tprimal_sig,Trule}(
        mi::Core.MethodInstance, debug_mode::Bool
    ) where {Tprimal_sig,Trule}
        return new{Tprimal_sig,Trule}(debug_mode, mi)
    end
end

_copy(x::P) where {P<:LazyFRule} = P(x.mi, x.debug_mode)

@inline function (rule::LazyFRule)(args::Vararg{Any,N}) where {N}
    return isdefined(rule, :rule) ? rule.rule(args...) : _build_rule!(rule, args)
end

@noinline function _build_rule!(rule::LazyFRule{sig,Trule}, args) where {sig,Trule}
    rule.rule = build_frule(get_interpreter(), rule.mi; debug_mode=rule.debug_mode)
    return rule.rule(args...)
end

function frule_type(interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode) where {C}
    if is_primitive(C, _get_sig(sig_or_mi))
        return debug_mode ? DebugFRule{typeof(frule!!)} : typeof(frule!!)
    end
    ir, _ = lookup_ir(interp, sig_or_mi)
    arg_types = map(CC.widenconst, ir.argtypes)
    fwd_args_type = Tuple{map(dual_type, arg_types)...}
    fwd_return_type = dual_type(Base.Experimental.compute_ir_rettype(ir))
    closure_type = RuleMC{fwd_args_type,fwd_return_type}
    Tderived_rule = DerivedFRule{closure_type}
    return debug_mode ? DebugFRule{Tderived_rule} : Tderived_rule
end

struct DynamicFRule{V}
    cache::V
    debug_mode::Bool
end

DynamicFRule(debug_mode::Bool) = DynamicFRule(Dict{Any,Any}(), debug_mode)

_copy(x::P) where {P<:DynamicFRule} = P(Dict{Any,Any}(), x.debug_mode)

function (dynamic_rule::DynamicFRule)(args::Vararg{Any,N}) where {N}
    args_dual = map(_dual, args)  # TODO: don't turn everything into a Dual, be clever with Argument and SSAValue
    sig = Tuple{map(_typeof ∘ primal, args_dual)...}
    rule = get(dynamic_rule.cache, sig, nothing)
    if rule === nothing
        rule = build_frule(get_interpreter(), sig; debug_mode=dynamic_rule.debug_mode)
        dynamic_rule.cache[sig] = rule
    end
    return rule(args_dual...)
end
