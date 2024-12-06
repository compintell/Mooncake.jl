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
    _is_primitive(C, sig_or_mi) && return (debug_mode ? DebugFRule(frule!!) : frule!!)

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
            dual_ir.argtypes[a] = Dual  # TODO: improve
        end
    end
    pushfirst!(dual_ir.argtypes, Any)

    # Modify dual IR incrementally
    dual_ir_comp = CC.IncrementalCompact(dual_ir)
    for ((_, i), inst) in dual_ir_comp
        modify_fwd_ad_stmts!(dual_ir_comp, primal_ir, interp, inst, i; debug_mode)
    end
    dual_ir_comp = CC.finish(dual_ir_comp)
    dual_ir_comp = CC.compact!(dual_ir_comp)

    CC.verify_ir(dual_ir_comp)

    # Optimize dual IR
    opt_dual_ir = optimise_ir!(dual_ir_comp; do_inline=false)  # TODO: toggle
    # @info "Inferred dual IR"
    # display(opt_dual_ir)  # TODO: toggle
    return opt_dual_ir
end

## Modification

function modify_fwd_ad_stmts!(
    dual_ir::CC.IncrementalCompact,
    primal_ir::IRCode,
    ::MooncakeInterpreter,
    stmt::Nothing,
    i::Integer;
    kwargs...,
)
    return nothing
end

function modify_fwd_ad_stmts!(
    dual_ir::CC.IncrementalCompact,
    primal_ir::IRCode,
    ::MooncakeInterpreter,
    stmt::GotoNode,
    i::Integer;
    kwargs...,
)
    return nothing
end

function modify_fwd_ad_stmts!(
    dual_ir::CC.IncrementalCompact,
    primal_ir::IRCode,
    ::MooncakeInterpreter,
    stmt::Core.GotoIfNot,
    i::Integer;
    kwargs...,
)
    # replace GotoIfNot with the call to primal
    Mooncake.replace_call!(dual_ir, CC.SSAValue(i), Expr(:call, _primal, stmt.cond))
    # reinsert the GotoIfNot right after the call to primal
    # (incremental insertion cannot be done before "where we are")
    new_gotoifnot_inst = CC.NewInstruction(
        Core.GotoIfNot(CC.SSAValue(i), stmt.dest),  #
        Any,
        CC.NoCallInfo(),
        Int32(1),  # meaningless
        CC.IR_FLAG_REFINED,
    )
    # stick the new instruction in the previous CFG block
    reverse_affinity = true
    CC.insert_node_here!(dual_ir, new_gotoifnot_inst, reverse_affinity)
    return nothing
end

# TODO: wrapping in Dual must not be systematic (e.g. Argument or SSAValue)
_frule!!_makedual(f, args::Vararg{Any,N}) where {N} = frule!!(make_dual.((f, args...))...)

struct DynamicFRule{V}
    cache::V
    debug_mode::Bool
end

DynamicFRule(debug_mode::Bool) = DynamicFRule(Dict{Any,Any}(), debug_mode)

_copy(x::P) where {P<:DynamicFRule} = P(Dict{Any,Any}(), x.debug_mode)

function (dynamic_rule::DynamicFRule)(args::Vararg{Any,N}) where {N}
    args_dual = map(make_dual, args)  # TODO: don't turn everything into a Dual, be clever with Argument and SSAValue
    sig = Tuple{map(_typeof ∘ primal, args_dual)...}
    rule = get(dynamic_rule.cache, sig, nothing)
    if rule === nothing
        rule = build_frule(get_interpreter(), sig; debug_mode=dynamic_rule.debug_mode)
        dynamic_rule.cache[sig] = rule
    end
    return rule(args_dual...)
end

function modify_fwd_ad_stmts!(
    dual_ir::CC.IncrementalCompact,
    primal_ir::IRCode,
    ::MooncakeInterpreter,
    stmt::ReturnNode,
    i::Integer;
    kwargs...,
)
    # the return node becomes a Dual so it changes type
    # flag to re-run type inference
    dual_ir[SSAValue(i)][:type] = Any
    dual_ir[SSAValue(i)][:flag] = CC.IR_FLAG_REFINED
    return nothing
end

function modify_fwd_ad_stmts!(
    dual_ir::CC.IncrementalCompact,
    primal_ir::IRCode,
    ::MooncakeInterpreter,
    stmt::PhiNode,
    i::Integer;
    kwargs...,
)
    dual_ir[SSAValue(i)][:stmt] = inc_args(stmt)  # TODO: translate constants into constant Duals
    dual_ir[SSAValue(i)][:type] = Any
    dual_ir[SSAValue(i)][:flag] = CC.IR_FLAG_REFINED
    return nothing
end

function modify_fwd_ad_stmts!(
    dual_ir::CC.IncrementalCompact,
    primal_ir::IRCode,
    interp::MooncakeInterpreter,
    stmt::Expr,
    i::Integer;
    debug_mode,
)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        sig, mi = if isexpr(stmt, :invoke)
            mi = stmt.args[1]::Core.MethodInstance
            mi.specTypes, mi
        else
            sig_types = map(stmt.args) do a
                get_forward_primal_type(primal_ir, a)
            end
            Tuple{sig_types...}, missing
        end
        shifted_args = if isexpr(stmt, :invoke)
            inc_args(stmt).args[2:end]  # first arg is method instance
        else
            inc_args(stmt).args
        end
        if is_primitive(context_type(interp), sig)
            call_frule = Expr(:call, _frule!!_makedual, shifted_args...)
            replace_call!(dual_ir, SSAValue(i), call_frule)
        else
            if isexpr(stmt, :invoke)
                rule = build_frule(interp, mi; debug_mode)
            else
                @assert isexpr(stmt, :call)
                rule = DynamicFRule(debug_mode)
            end
            # TODO: could this insertion of a naked rule in the IR cause a memory leak?
            call_rule = Expr(:call, rule, shifted_args...)
            replace_call!(dual_ir, SSAValue(i), call_rule)
        end
    elseif isexpr(stmt, :boundscheck)
        nothing
    elseif isexpr(stmt, :code_coverage_effect)
        replace_call!(dual_ir, SSAValue(i), nothing)
    else
        throw(
            ArgumentError(
                "Expressions of type `:$(stmt.head)` are not yet supported in forward mode"
            ),
        )
    end
end

get_forward_primal_type(ir::IRCode, a::Argument) = ir.argtypes[a.n]
get_forward_primal_type(ir::IRCode, ssa::SSAValue) = ir[ssa][:type]
get_forward_primal_type(::IRCode, x::QuoteNode) = _typeof(x.value)
get_forward_primal_type(::IRCode, x) = _typeof(x)
function get_forward_primal_type(::IRCode, x::GlobalRef)
    return isconst(x) ? _typeof(getglobal(x.mod, x.name)) : x.binding.ty
end
function get_forward_primal_type(::IRCode, x::Expr)
    x.head === :boundscheck && return Bool
    return error("Unrecognised expression $x found in argument slot.")
end
