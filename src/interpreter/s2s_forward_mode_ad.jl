function build_frule(args...; debug_mode=false)
    interp = get_interpreter()
    sig = _typeof(TestUtils.__get_primals(args))
    return build_frule(interp, sig; debug_mode)
end

function build_frule(
    interp::MooncakeInterpreter{C},
    sig_or_mi;
    debug_mode=false,
    silence_debug_messages=true,
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
            return _copy(interp.oc_cache[oc_cache_key])
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

function generate_forward_ir(
    interp::MooncakeInterpreter,
    sig_or_mi;
    debug_mode=false,
    do_inline=true,
)
    # Reset id count. This ensures that the IDs generated are the same each time this
    # function runs.
    seed_id!()

    # Grab code associated to the primal.
    primal_ir, _ = lookup_ir(interp, sig_or_mi)

    # Normalise the IR.
    isva, spnames = is_vararg_and_sparam_names(sig_or_mi)
    ir = normalise!(primal_ir, spnames)

    fwd_ir = dualize_ir(ir)
    opt_fwd_ir = optimise_ir!(fwd_ir; do_inline)
    return opt_fwd_ir
end

function dualize_ir(ir::IRCode)
    new_stmts_stmt = map(make_fwd_ad_stmt, ir.stmts.stmt)
    new_stmts_type = fill(Any, length(ir.stmts.type))
    new_stmts_info = ir.stmts.info
    new_stmts_line = ir.stmts.line
    new_stmts_flag = fill(CC.IR_FLAG_REFINED, length(ir.stmts.flag))
    new_stmts = CC.InstructionStream(
        new_stmts_stmt,
        new_stmts_type,
        new_stmts_info,
        new_stmts_line,
        new_stmts_flag,
    )
    new_cfg = ir.cfg
    new_linetable = ir.linetable
    rule_type = Any
    new_argtypes = convert(Vector{Any}, vcat(rule_type, map(make_fwd_argtype, ir.argtypes)))
    new_meta = ir.meta
    new_sptypes = ir.sptypes
    return IRCode(new_stmts, new_cfg, new_linetable, new_argtypes, new_meta, new_sptypes)
end

make_fwd_argtype(::Type{P}) where {P} = dual_type(P)
make_fwd_argtype(c::Core.Const) = Dual  # TODO: refine to type of const

function make_fwd_ad_stmt(stmt::Expr)
    interp = get_interpreter()  # TODO: pass it around
    C = context_type(interp)
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        mi = stmt.args[1]::Core.MethodInstance
        sig = mi.specTypes
        if is_primitive(C, sig)
            shifted_args = inc_args(stmt.args)
            new_stmt = Expr(
                :call,
                :($frule!!),
                stmt.args[2],
                shifted_args[3:end]...
            )
            return new_stmt
        else
            throw(ArgumentError("Recursing into non-primitive calls is not yet supported in forward mode"))
        end
        return stmt
    else
        throw(ArgumentError("Expressions of type `:$(stmt.head)` are not yet supported in forward mode"))
    end
end

function make_fwd_ad_stmt(stmt::ReturnNode)
    return stmt
end

struct DerivedFRule{Tfwd_oc}
    fwd_oc::Tfwd_oc
end

_copy(rule::DerivedFRule) = deepcopy(rule)

@inline function (fwd::DerivedFRule)(args::Vararg{Dual,N}) where {N}
    return fwd.fwd_oc.oc(args...)
end
