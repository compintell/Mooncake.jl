"""
    ircode(
        inst::Vector{Any},
        argtypes::Vector{Any},
        sptypes::Vector{CC.VarState}=CC.VarState[],
    ) -> IRCode

Constructs an instance of an `IRCode`. This is useful for constructing test cases with known
properties.

No optimisations or type inference are performed on the resulting `IRCode`, so that
the `IRCode` contains exactly what is intended by the caller. Please make use of
`infer_types!` if you require the types to be inferred.

Edges in `PhiNode`s, `GotoIfNot`s, and `GotoNode`s found in `inst` must refer to lines (as
in `CodeInfo`). In the `IRCode` returned by this function, these line references are
translated into block references.
"""
function ircode(
    insts::Vector{Any}, argtypes::Vector{Any}, sptypes::Vector{CC.VarState}=CC.VarState[]
)
    cfg = CC.compute_basic_blocks(insts)
    insts = __line_numbers_to_block_numbers!(insts, cfg)
    stmts = __insts_to_instruction_stream(insts)
    linetable = [CC.LineInfoNode(Tapir, :ircode, :ir_utils, Int32(1), Int32(0))]
    meta = Expr[]
    return CC.IRCode(stmts, cfg, linetable, argtypes, meta, CC.VarState[])
end

#=
Converts any edges in `GotoNode`s, `GotoIfNot`s, `PhiNode`s, and `:enter` expressions which
refer to line numbers into references to block numbers. The `cfg` provides the information
required to perform this conversion.

For context, `CodeInfo` objects have references to line numbers, while `IRCode` uses
block numbers.

This code is copied over directly from the body of `Core.Compiler.inflate_ir!`.
=#
function __line_numbers_to_block_numbers!(insts::Vector{Any}, cfg::CC.CFG)
    for i in eachindex(insts)
        stmt = insts[i]
        if isa(stmt, GotoNode)
            insts[i] = GotoNode(CC.block_for_inst(cfg, stmt.label))
        elseif isa(stmt, GotoIfNot)
            insts[i] = GotoIfNot(stmt.cond, CC.block_for_inst(cfg, stmt.dest))
        elseif isa(stmt, PhiNode)
            insts[i] = PhiNode(
                Int32[CC.block_for_inst(cfg, Int(edge)) for edge in stmt.edges], stmt.values
            )
        elseif Meta.isexpr(stmt, :enter)
            stmt.args[1] = CC.block_for_inst(cfg, stmt.args[1]::Int)
            insts[i] = stmt
        end
    end
    return insts
end

#=
Produces an instruction stream whose
- `inst` field is `insts`,
- `type` field is all `Any`,
- `info` field is all `Core.Compiler.NoCallInfo`,
- `line` field is all `Int32(1)`, and
- `flag` field is all `Core.Compiler.IR_FLAG_REFINED`.

As such, if you wish to ensure that your `IRCode` prints nicely, you should ensure that its
linetable field has at least one element.
=#
function __insts_to_instruction_stream(insts::Vector{Any})
    return CC.InstructionStream(
        insts,
        fill(Any, length(insts)),
        fill(CC.NoCallInfo(), length(insts)),
        fill(Int32(1), length(insts)),
        fill(CC.IR_FLAG_REFINED, length(insts)),
    )
end

"""
    infer_ir!(ir::IRCode) -> IRCode

Runs type inference on `ir`, which mutates `ir`, and returns it.

Note: the compiler will not infer the types of anything where the corrsponding element of
`ir.stmts.flag` is not set to `Core.Compiler.IR_FLAG_REFINED`. Nor will it attempt to refine
the type of the value returned by a `:invoke` expressions. Consequently, if you find that
the types in your IR are not being refined, you may wish to check that neither of these
things are happening.
"""
function infer_ir!(ir::IRCode)
    return __infer_ir!(ir, CC.NativeInterpreter(), __get_toplevel_mi_from_ir(ir, Tapir))
end

# Given some IR, generates a MethodInstance suitable for passing to infer_ir!, if you don't
# already have one with the right argument types. Credit to @oxinabox:
# https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802#file-example-jl-L54
function __get_toplevel_mi_from_ir(ir, _module::Module)
    mi = ccall(:jl_new_method_instance_uninit, Ref{Core.MethodInstance}, ());
    mi.specTypes = Tuple{map(_type, ir.argtypes)...}
    mi.def = _module
    return mi
end

# Run type inference and constant propagation on the ir. Credit to @oxinabox:
# https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802#file-example-jl-L54
function __infer_ir!(ir, interp::CC.AbstractInterpreter, mi::CC.MethodInstance)
    method_info = CC.MethodInfo(#=propagate_inbounds=#true, nothing)
    min_world = world = CC.get_world_counter(interp)
    max_world = Base.get_world_counter()
    irsv = CC.IRInterpretationState(
        interp, method_info, ir, mi, ir.argtypes, world, min_world, max_world
    )
    rt = CC._ir_abstract_constant_propagation(interp, irsv)
    return ir
end

# In automatically generated code, it is meaningless to include code coverage effects.
# Moreover, it seems to cause some serious inference probems. Consequently, it makes sense
# to remove such effects before optimising IRCode.
function __strip_coverage!(ir::IRCode)
    for n in eachindex(ir.stmts.inst)
        if Meta.isexpr(ir.stmts.inst[n], :code_coverage_effect)
            ir.stmts.inst[n] = nothing
        end
    end
    return ir
end

"""
    optimise_ir!(ir::IRCode, show_ir=false)

Run a fairly standard optimisation pass on `ir`. If `show_ir` is `true`, displays the IR
to `stdout` at various points in the pipeline -- this is sometimes useful for debugging.
"""
function optimise_ir!(ir::IRCode; show_ir=false, do_inline=true)
    if show_ir
        println("Pre-optimization")
        display(ir)
        println()
    end
    CC.verify_ir(ir)
    ir = __strip_coverage!(ir)
    ir = CC.compact!(ir)
    local_interp = CC.NativeInterpreter()
    mi = __get_toplevel_mi_from_ir(ir, @__MODULE__);
    ir = __infer_ir!(ir, local_interp, mi)
    if show_ir
        println("Post-inference")
        display(ir)
        println()
    end
    inline_state = CC.InliningState(local_interp)
    CC.verify_ir(ir)
    if do_inline
        ir = CC.ssa_inlining_pass!(ir, inline_state, #=propagate_inbounds=#true)
        ir = CC.compact!(ir)
    end
    ir = __strip_coverage!(ir)
    ir = CC.sroa_pass!(ir, inline_state)
    ir = CC.adce_pass!(ir, inline_state)
    ir = CC.compact!(ir)
    # CC.verify_ir(ir, true, false, CC.optimizer_lattice(local_interp))
    CC.verify_linetable(ir.linetable, true)
    if show_ir
        println("Post-optimization")
        display(ir)
        println()
    end
    return ir
end

"""
    lookup_ir(interp::AbstractInterpreter, sig::Type{<:Tuple})::Tuple{IRCode, T}

Get the IR associated to `sig` under `interp`. Throws `ArgumentError`s if there is
no code found, or if more than one `IRCode` instance returned.

Returns a tuple containing the `IRCode` and its return type.
"""
function lookup_ir(interp::CC.AbstractInterpreter, sig::Type{<:Tuple})

    # If `invoke` is the function in question, then do something complicated.
    if sig.parameters[1] == typeof(invoke)
        return lookup_invoke_ir(interp, sig)
    end

    # Look up the `IRCode` using the standard mechanism.
    output = Base.code_ircode_by_type(sig; interp)
    if isempty(output)
        throw(ArgumentError("No methods found for signature $sig"))
    elseif length(output) > 1
        throw(ArgumentError("$(length(output)) methods found for signature $sig"))
    end
    return only(output)[1]
end

is_invoke_sig(sig) = sig.parameters[1] == typeof(invoke)

function static_sig(sig)
    ps = sig.parameters
    return is_invoke_sig(sig) ? Tuple{ps[2], ps[3].parameters[1].parameters...} : sig
end

"""
    lookup_invoke_ir(interp::CC.AbstractInterpreter, sig::Type{<:Tuple})

Looks up the `IRCode` associated to an `invoke` call. If the first parameter of `sig` is not
`typeof(invoke)` then an `ArgumentError` is thrown. No other error checking is performed.
"""
function lookup_invoke_ir(interp::CC.AbstractInterpreter, sig::Type{<:Tuple})
    if sig.parameters[1] != typeof(invoke)
        throw(ArgumentError("Expected signature for a call to `invoke`."))
    end

    # Construct the static signature, and dynamic signature.
    ps = sig.parameters
    _sig = static_sig(sig)
    dynamic_sig = Tuple{ps[2], ps[4:end]...}

    # Lookup all methods which could apply to the types provided in the signature, and pick
    # the one which `which` says would get applied.
    # Base on https://github.com/JuliaLang/julia/blob/v1.10.4/base/reflection.jl#L1485
    matches = Base._methods_by_ftype(_sig, #=lim=#-1, Base.get_world_counter())
    m = which(_sig)
    match = only(filter(_m -> m === _m.method, matches))
    meth = Base.func_for_method_checked(match.method, _sig, match.sparams)
    (code, _) = Core.Compiler.typeinf_ircode(
        interp, meth, dynamic_sig, match.sparams, #=optimize_until=#nothing
    )
    return code
end

"""
    is_reachable_return_node(x::ReturnNode)

Determine whether `x` is a `ReturnNode`, and if it is, if it is also reachable. This is
purely a function of whether or not its `val` field is defined or not.
"""
is_reachable_return_node(x::ReturnNode) = isdefined(x, :val)
is_reachable_return_node(x) = false

"""
    is_unreachable_return_node(x::ReturnNode)

Determine whehter `x` is a `ReturnNode`, and if it is, if it is also unreachable. This is
purely a function of whether or not its `val` field is defined or not.
"""
is_unreachable_return_node(x::ReturnNode) = !isdefined(x, :val)
is_unreachable_return_node(x) = false

"""
    UnhandledLanguageFeatureException(message::String)

An exception used to indicate that some aspect of the Julia language which AD cannot handle
has been encountered.
"""
struct UnhandledLanguageFeatureException <: Exception
    msg::String
end

"""
    unhandled_feature(msg::String)

Throw an `UnhandledLanguageFeatureException` with message `msg`.
"""
unhandled_feature(msg::String) = throw(UnhandledLanguageFeatureException(msg))

"""
    inc_args(stmt)

Increment by `1` the `n` field of any `Argument`s present in `stmt`.
"""
inc_args(x::Expr) = Expr(x.head, map(__inc, x.args)...)
inc_args(x::ReturnNode) = isdefined(x, :val) ? ReturnNode(__inc(x.val)) : x
inc_args(x::IDGotoIfNot) = IDGotoIfNot(__inc(x.cond), x.dest)
inc_args(x::IDGotoNode) = x
function inc_args(x::IDPhiNode)
    new_values = Vector{Any}(undef, length(x.values))
    for n in eachindex(x.values)
        if isassigned(x.values, n)
            new_values[n] = __inc(x.values[n])
        end
    end
    return IDPhiNode(x.edges, new_values)
end
inc_args(::Nothing) = nothing
inc_args(x::GlobalRef) = x

__inc(x::Argument) = Argument(x.n + 1)
__inc(x) = x

"""
    new_inst(stmt, type=Any, flag=CC.IR_FLAG_REFINED)::NewInstruction

Create a `NewInstruction` with fields:
- `stmt` = `stmt`
- `type` = `type`
- `info` = `CC.NoCallInfo()`
- `line` = `Int32(1)`
- `flag` = `flag`
"""
function new_inst(@nospecialize(stmt), @nospecialize(type)=Any, flag=CC.IR_FLAG_REFINED)
    return NewInstruction(stmt, type, CC.NoCallInfo(), Int32(1), flag)
end
