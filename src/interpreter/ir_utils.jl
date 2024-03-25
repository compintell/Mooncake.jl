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
    linetable = [CC.LineInfoNode(Phi, :ircode, :ir_utils, Int32(1), Int32(0))]
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
    return __infer_ir!(ir, CC.NativeInterpreter(), __get_toplevel_mi_from_ir(ir, Phi))
end

# Sometimes types in `IRCode` have been replaced by constants, or partially-completed
# structs. A particularly common case is that the first element of the `argtypes` field of
# an `IRCode` is a `Core.Const` containing the function to be called. `_get_type` recovers
# the type in such situations.
_get_type(x::Core.PartialStruct) = x.typ
_get_type(x::Core.Const) = _typeof(x.val)
_get_type(T) = T

# Given some IR, generates a MethodInstance suitable for passing to infer_ir!, if you don't
# already have one with the right argument types. Credit to @oxinabox:
# https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802#file-example-jl-L54
function __get_toplevel_mi_from_ir(ir, _module::Module)
    mi = ccall(:jl_new_method_instance_uninit, Ref{Core.MethodInstance}, ());
    mi.specTypes = Tuple{map(_get_type, ir.argtypes)...}
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
    replace_all_uses_with!(ir::IRCode, value::SSAValue, new_value::Any) -> IRCode

Wherever `value` appears in `ir`, replace it with `new_value`.

Note: this will *not* effect anything in the `new_nodes` field of `ir`.
"""
function replace_all_uses_with!(ir::IRCode, value::SSAValue, new_value::Any)
    insts = ir.stmts.inst
    for (n, inst) in enumerate(insts)
        insts[n] = replace_uses_with(inst, value, new_value)
    end
    return ir
end

#=
    replace_uses_with(x::Any, value::SSAValue, new_value::Any)

Replace all occurences of `value` in the IR node `x` with `new_value`. The semantics of this
are node-dependent.

If `value` appears by itself as a constant, it will not be replaced.
=#
replace_uses_with(x::Any, ::SSAValue, _) = x # constants
function replace_uses_with(x::Expr, v::SSAValue, new_v)
    return Expr(x.head, [_replace(v, new_v, a) for a in x.args]...)
end
replace_uses_with(x::GotoNode, ::SSAValue, _) = x
function replace_uses_with(x::GotoIfNot, v::SSAValue, new_v)
    return GotoIfNot(_replace(v, new_v, x.cond), x.dest)
end
function replace_uses_with(x::PhiNode, v::SSAValue, new_v)
    values = x.values
    new_values = Vector{Any}(undef, length(values))
    for n in eachindex(new_values)
        if isassigned(values, n)
            new_values[n] = _replace(v, new_v, values[n])
        end
    end
    return PhiNode(x.edges, new_values)
end
replace_uses_with(x::PiNode, v::SSAValue, new_v) = PiNode(_replace(v, new_v, x.val), x.typ)
replace_uses_with(x::QuoteNode, ::SSAValue, _) = x
function replace_uses_with(x::ReturnNode, v::SSAValue, new_v)
    return isdefined(x, :val) ? ReturnNode(_replace(v, new_v, x.val)) : x
end

# Return new_value if val equals current_val.
_replace(val::SSAValue, new_val, current_val) = val == current_val ? new_val : current_val

"""
    lookup_ir(interp::AbstractInterpreter, sig::Type{<:Tuple})::Tuple{IRCode, T}

Get the IR unique IR associated to `sig` under `interp`. Throws `ArgumentError`s if there is
no code found, or if more than one `IRCode` instance returned.

Returns a tuple containing the `IRCode` and its return type.
"""
function lookup_ir(interp::CC.AbstractInterpreter, sig::Type{<:Tuple})
    output = Base.code_ircode_by_type(sig; interp)
    if isempty(output)
        throw(ArgumentError("No methods found for signature $sig"))
    elseif length(output) > 1
        throw(ArgumentError("$(length(output)) methods found for signature $sig"))
    end
    return only(output)
end

"""
    is_reachable(x::ReturnNode)

Determine whether `x` is a `ReturnNode`, and if it is, if it is also reachable. This is
purely a function of whether or not its `val` field is defined or not.
"""
is_reachable_return_node(x::ReturnNode) = isdefined(x, :val)
is_reachable_return_node(x) = false

"""
    globalref_type(x::GlobaRef)

Returns the static type of the value referred to by `x`.
"""
globalref_type(x::GlobalRef) = isconst(x) ? _typeof(getglobal(x.mod, x.name)) : x.binding.ty

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
