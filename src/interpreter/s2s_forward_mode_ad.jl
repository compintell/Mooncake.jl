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
        if haskey(interp.oc_cache, oc_cache_key)
            return interp.oc_cache[oc_cache_key]
        else
            # Derive forward-pass IR, and shove in a `MistyClosure`.
            forward_ir = generate_forward_ir(interp, sig_or_mi; debug_mode)
            fwd_oc = MistyClosure(forward_ir; do_compile=true)
            raw_rule = DerivedFRule(fwd_oc)
            rule = debug_mode ? DebugFRule(raw_rule) : raw_rule
            interp.oc_cache[oc_cache_key] = rule
            return rule
        end
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

function generate_forward_ir(
    interp::MooncakeInterpreter, sig_or_mi; debug_mode=false, do_inline=true
)
    # Reset id count. This ensures that the IDs generated are the same each time this
    # function runs.
    seed_id!()

    # Grab code associated to the primal.
    primal_ir, _ = lookup_ir(interp, sig_or_mi)

    # Normalise the IR.
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    ir = normalise!(primal_ir, spnames)

    # Differentiate the IR
    fwd_ir = copy(ir)
    for i in 1:length(ir.stmts)
        # betting on the fact that lines don't change before compact! is called, even with insertions
        stmt = fwd_ir[SSAValue(i)][:stmt]
        make_fwd_ad_stmts!(fwd_ir, ir, interp, stmt, i; debug_mode)
    end
    pushfirst!(fwd_ir.argtypes, Any)  # the rule will be the first argument
    fwd_ir_compact = CC.compact!(fwd_ir)

    # Optimize the IR
    opt_fwd_ir = optimise_ir!(fwd_ir_compact; do_inline)
    return opt_fwd_ir
end

"""
    make_fwd_ad_stmts!(ir, stmt, i)

Modify `ir` in-place to transform statement `stmt`, originally at position `SSAValue(i)`, into one or more derivative statements (which get inserted).
"""
function make_fwd_ad_stmts! end

function make_fwd_ad_stmts!(
    fwd_ir::IRCode,
    ir::IRCode,
    ::MooncakeInterpreter,
    stmt::ReturnNode,
    i::Integer;
    kwargs...,
)
    inst = fwd_ir[SSAValue(i)]
    # the return node becomes a Dual so it changes type
    # flag to re-run type inference
    inst[:type] = Any
    inst[:flag] = CC.IR_FLAG_REFINED
    return nothing
end

function make_fwd_ad_stmts!(
    fwd_ir::IRCode,
    ir::IRCode,
    interp::MooncakeInterpreter,
    stmt::Expr,
    i::Integer;
    debug_mode,
)
    inst = fwd_ir[SSAValue(i)]
    C = context_type(interp)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        sig, mi = if isexpr(stmt, :invoke)
            mi = stmt.args[1]::Core.MethodInstance
            mi.specTypes, mi
        else
            sig_types = map(Base.Fix1(get_forward_primal_type, ir), stmt.args)
            Tuple{sig_types...}, missing
        end
        shifted_args = inc_args(stmt.args)
        if is_primitive(C, sig)
            inst[:stmt] = Expr(:call, frule!!, shifted_args[2:end]...)
            inst[:info] = CC.NoCallInfo()
            inst[:type] = Any
            inst[:flag] = CC.IR_FLAG_REFINED
        elseif isexpr(stmt, :invoke)
            rule = build_frule(interp, mi; debug_mode)
            # modify the original statement to use `rule`
            inst[:stmt] = Expr(:call, rule, shifted_args[2:end]...)
            inst[:info] = CC.NoCallInfo()
            inst[:type] = Any
            inst[:flag] = CC.IR_FLAG_REFINED
        elseif isexpr(stmt, :call)
            throw(
                ArgumentError("Expressions of type `:call` not supported in forward mode")
            )
        end
    else
        throw(
            ArgumentError(
                "Expressions of type `:$(stmt.head)` are not yet supported in forward mode"
            ),
        )
    end
end

get_forward_primal_type(ir::IRCode, a::Argument) = ir.arg_types[a]
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
