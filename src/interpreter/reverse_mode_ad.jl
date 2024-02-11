# FwdsInsts have the same signature as Insts, but have different side-effects.
const FwdsInst = Core.OpaqueClosure{Tuple{Int}, Int}

# The backwards instructions don't actually need to return an Int, however, there is
# currently a performance bug in OpaqueClosures which means that an allocation is produced
# if a constant is returned. Consequently, we have to return something non-constant.
# See https://github.com/JuliaLang/julia/issues/52620 for info.
# By convention, any backwards instruction which "does nothing" just returns its input, and
# does no other work.
const BwdsInst = Core.OpaqueClosure{Tuple{Int}, Int}

const CoDualSlot{V} = AbstractSlot{V} where {V<:CoDual}

primal_eltype(::CoDualSlot{CoDual{P, T}}) where {P, T} = P

mutable struct FwdStack{V<:CoDual, P, Tstack<:Stack} <: AbstractSlot{V}
    primal_slot::P
    tangent_stack::Tstack
    function FwdStack{CoDual{P, T}}() where {P, T}
        return new{codual_type(P), P, Stack{T}}(P(), Stack{T}())
    end
    FwdStack{CoDual}() = FwdStack{CoDual{Any, Any}}()
    FwdStack(::Pslot) where {P, Pslot<:SlotRef{P}} = FwdStack{codual_type(P)}()
end

Base.isassigned(x::FwdStack) = isassigned(x.tangent_stack)

const CoDualStack{V} = Union{ConstSlot{V}, FwdStack{V}} where {V<:CoDual}

function FwdStack(x::C) where {C<:CoDual}
    stack = FwdStack{C}()
    push!(stack, x)
    return stack
end

Base.getindex(x::FwdStack{V}) where {V<:CoDual} = V(x.primal_slot[], x.tangent_stack[])

function Base.push!(x::FwdStack{V}, v::V) where {V<:CoDual}
    x.primal_slot[] = primal(v)
    push!(x.tangent_stack, tangent(v))
    return nothing
end

make_codual_stack(x::SlotRef{P}) where {P} = FwdStack(x)
function make_codual_stack(x::ConstSlot{P}) where {P} # REVISIT THIS TO USE CONSTSLOT AGAIN
    stack = FwdStack{codual_type(P)}()
    push!(stack, uninit_codual(x[]))
    return stack
end

# Operations on Slots involving CoDuals

function increment_tangent!(x::CoDualSlot{C}, t) where {C}
    x_val = x[]
    x[] = C(primal(x_val), increment!!(tangent(x_val), t))
    return nothing
end

function increment_tangent!(x::FwdStack{V}, t) where {V}
    x.tangent_stack[] = increment!!(x.tangent_stack[], t)
    return nothing
end

## ReturnNode
function build_coinsts(node::ReturnNode, _, _rrule!!, ::Int, ::Int, ::Bool)
    return build_coinsts(ReturnNode, _rrule!!.return_slot, _get_slot(node.val, _rrule!!))
end
function build_coinsts(::Type{ReturnNode}, ret_slot::SlotRef{<:CoDual}, val::CoDualStack)
    fwds_inst = build_inst(ReturnNode, ret_slot, val)
    bwds_inst = @opaque (j::Int) -> (increment_tangent!(val, tangent(ret_slot[])); return j)
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## GotoNode
build_coinsts(x::GotoNode, _, _, ::Int, ::Int, ::Bool) = build_coinsts(GotoNode, x.label)
function build_coinsts(::Type{GotoNode}, dest::Int)
    return build_inst(GotoNode, dest)::FwdsInst, (@opaque (j::Int) -> j)::BwdsInst
end

## GotoIfNot
function build_coinsts(x::GotoIfNot, _, _rrule!!, ::Int, b::Int, is_blk_end::Bool)
    return build_coinsts(GotoIfNot, x.dest, b + 1, _get_slot(x.cond, _rrule!!))
end
function build_coinsts(::Type{GotoIfNot}, dest::Int, next_blk::Int, cond::CoDualStack)
    fwds_inst = @opaque (p::Int) -> primal(cond[]) ? next_blk : dest
    bwds_inst = @opaque (j::Int) -> j
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## PhiNode

function build_coinsts(ir_insts::Vector{PhiNode}, _, _rrule!!, n_first::Int, b::Int, is_blk_end::Bool)
    nodes = (build_typed_phi_nodes(ir_insts, _rrule!!, n_first)..., )
    next_blk = _standard_next_block(is_blk_end, b)
    return build_coinsts(Vector{PhiNode}, nodes, next_blk, Stack{Int}())
end

function build_coinsts(
    ::Type{Vector{PhiNode}}, nodes::NTuple{N, TypedPhiNode}, next_blk::Int,
) where {N}

    # Check that we're operating on CoDuals.
    @assert all(x -> x.ret_slot isa CoDualSlot, nodes)
    @assert all(x -> all(y -> isa(y, CoDualSlot), x.values), nodes)

    # Construct instructions.
    fwds_inst = @opaque function (p::Int)
        map(Base.Fix2(store_tmp_value!, p), nodes) # transfer new value into tmp slots
        map(Base.Fix2(transfer_tmp_value!, p), nodes) # transfer new value from tmp slots into ret slots
        return next_blk
    end
    bwds_inst = @opaque (j::Int) -> j
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

function transfer_tmp_value!(x::TypedPhiNode{<:FwdStack}, prev_blk::Int)
    map(x.edges, x.values) do edge, v
        (edge == prev_blk) && isassigned(v) && (push!(x.ret_slot, x.tmp_slot[]))
    end
    return nothing
end

## PiNode
function build_coinsts(x::PiNode, _, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
    val = _get_slot(x.val, _rrule!!)
    ret = _rrule!!.slots[n]
    return build_coinsts(PiNode, val, ret, _standard_next_block(is_blk_end, b))
end
function build_coinsts(
    ::Type{PiNode}, val::CoDualStack, ret::CoDualStack{R}, next_blk::Int
) where {R}
    make_fwds(v) = R(primal(v), tangent(v))
    fwds_inst = @opaque function (p::Int)
        push!(ret, make_fwds(val[]))
        return next_blk
    end
    bwds_inst = @opaque function (j::Int)
        increment_tangent!(val, pop!(ret.tangent_stack))
        return j
    end
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## GlobalRef
function build_coinsts(x::GlobalRef, _, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
    next_blk = _standard_next_block(is_blk_end, b)
    return build_coinsts(GlobalRef, _globalref_to_slot(x), _rrule!!.slots[n], next_blk)
end
function build_coinsts(::Type{GlobalRef}, x::AbstractSlot, out::CoDualStack, next_blk::Int)
    fwds_inst = @opaque (p::Int) -> (push!(out, uninit_codual(x[])); return next_blk)
    bwds_inst = @opaque (j::Int) -> (pop!(out.tangent_stack); return j)
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## QuoteNode and literals
function build_coinsts(node, _, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
    x = ConstSlot(zero_codual(node isa QuoteNode ? node.value : node))
    next_blk = _standard_next_block(is_blk_end, b)
    return build_coinsts(nothing, x, _rrule!!.slots[n], next_blk)
end
function build_coinsts(::Nothing, x::ConstSlot, out::CoDualStack, next_blk::Int)
    fwds_inst = @opaque (p::Int) -> (push!(out, x[]); return next_blk)
    bwds_inst = @opaque (j::Int) -> (pop!(out.tangent_stack); return j)
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## Expr

get_rrule!!_evaluator(::typeof(_eval)) = rrule!!
get_rrule!!_evaluator(in_f::InterpretedFunction) = build_rrule!!(in_f)
get_rrule!!_evaluator(::DelayedInterpretedFunction) = rrule!!

# Constructs a Vector which can holds instances of the pullback associated to
# `__rrule!!` when applied to the types in `codual_sig`. If `__rrule!!` infers for these
# types, then we should get a concretely-typed containers. Conversely, if inference fails,
# we fallback to `Any`.
function build_pb_stack(__rrule!!, evaluator, arg_slots)
    deval = zero_codual(evaluator)
    codual_sig = Tuple{_typeof(deval), map(eltype, arg_slots)...}
    possible_output_types = Base.return_types(__rrule!!, codual_sig)
    if length(possible_output_types) == 0
        throw(error("No return type inferred for __rrule!! with sig $codual_sig"))
    elseif length(possible_output_types) > 1
        @warn "Too many output types inferred"
        display(possible_output_types)
        println()
        throw(error("> 1 return type inferred for __rrule!! with sig $codual_sig "))
    end
    T_pb!! = only(possible_output_types)
    if T_pb!! <: Tuple && T_pb!! !== Union{}
        pb_stack = Stack{T_pb!!.parameters[2]}()
    else
        pb_stack = Stack{Any}()
    end
    return pb_stack
end

function build_coinsts(ir_inst::Expr, in_f, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
    is_invoke = Meta.isexpr(ir_inst, :invoke)
    next_blk = _standard_next_block(is_blk_end, b)
    val_slot = _rrule!!.slots[n]
    if Meta.isexpr(ir_inst, :boundscheck)
        return build_coinsts(Val(:boundscheck), val_slot, next_blk)
    elseif is_invoke || Meta.isexpr(ir_inst, :call)

        # Extract args refs.
        __args = is_invoke ? ir_inst.args[2:end] : ir_inst.args
        arg_slots = map(arg -> _get_slot(arg, _rrule!!), (__args..., ))

        # if primal(arg_slots[1][]) isa typeof(_new_)
        #     println("in new")
        #     display(_typeof(arg_slots))
        #     println()
        #     display(map(getindex, arg_slots))
        #     println()
        #     display(__args)
        #     println()
        #     println("primal arg slots")
        #     display(map(arg -> _get_slot(arg, in_f), (__args..., )))
        #     println()
        #     println("tangent arg slots")
        #     display(arg_slots)
        #     println()
        # end

        # Construct signature, and determine how the rrule is to be computed.
        primal_sig = _typeof(map(primal ∘ getindex, arg_slots))
        evaluator = get_evaluator(in_f.ctx, primal_sig, in_f.interp, is_invoke)
        __rrule!! = get_rrule!!_evaluator(evaluator)

        # Create stack for storing pullbacks.
        pb_stack = build_pb_stack(__rrule!!, evaluator, arg_slots)

        return build_coinsts(
            Val(:call), val_slot, arg_slots, evaluator, __rrule!!, pb_stack, next_blk
        )
    elseif ir_inst.head in [
        :code_coverage_effect, :gc_preserve_begin, :gc_preserve_end, :loopinfo,
        :leave, :pop_exception,
    ]
        return build_coinsts(Val(:skipped_expression), next_blk)
    elseif Meta.isexpr(ir_inst, :throw_undef_if_not)
        slot_to_check = _get_slot(ir_inst.args[2], _rrule!!)
        return build_coinsts(Val(:throw_undef_if_not), slot_to_check, next_blk)
    else
        throw(error("Unrecognised expression $ir_inst"))
    end
end

function build_coinsts(::Val{:boundscheck}, out::CoDualStack, next_blk::Int)
    @assert primal_eltype(out) == Bool
    fwds_inst = @opaque (p::Int) -> (push!(out, zero_codual(true)); return next_blk)
    bwds_inst = @opaque (j::Int) -> (pop!(out.tangent_stack); return j)
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

function replace_tangent!(x::AbstractSlot{<:CoDual{Tx, Tdx}}, new_tangent::Tdx) where {Tx, Tdx}
    x[] = CoDual(primal(x[]), new_tangent)
    return nothing
end

function replace_tangent!(x::AbstractSlot{<:CoDual}, new_tangent)
    x[] = CoDual(primal(x[]), new_tangent)
    return nothing
end

replace_tangent!(::ConstSlot{<:CoDual}, new_tangent) = nothing

function replace_tangent!(x::FwdStack, new_tangent)
    x.tangent_stack[] = new_tangent
    return nothing
end

function build_coinsts(
    ::Val{:call},
    out::CoDualStack,
    arg_slots::NTuple{N, CoDualStack} where {N},
    evaluator::Teval,
    __rrule!!::Trrule!!,
    pb_stack::Stack,
    next_blk::Int,
) where {Teval, Trrule!!}

    function fwds_pass()
        args = tuple_map(getindex, arg_slots)
        _out, pb!! = __rrule!!(zero_codual(evaluator), args...)
        push!(out, _out)
        push!(pb_stack, pb!!)
        return nothing
    end
    fwds_inst = @opaque function (p::Int)
        fwds_pass()
        return next_blk
    end

    function bwds_pass()
        pb!! = pop!(pb_stack)
        dout = pop!(out.tangent_stack)
        dargs = tuple_map(set_immutable_to_zero ∘ tangent ∘ getindex, arg_slots)
        new_dargs = pb!!(dout, NoTangent(), dargs...)
        map(increment_tangent!, arg_slots, new_dargs[2:end])
        return nothing
    end
    bwds_inst = @opaque function (j::Int)
        bwds_pass()
        return j
    end
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

function build_coinsts(::Val{:skipped_expression}, next_blk::Int)
    return (@opaque (p::Int) -> next_blk), (@opaque (j::Int) -> j)
end

function build_coinsts(::Val{:throw_undef_if_not}, slot::AbstractSlot, next_blk::Int)
    fwds_inst::FwdsInst = @opaque function (prev_blk::Int)
        !isassigned(slot) && throw(error("Boooo, not assigned"))
        return next_blk
    end
    bwds_inst::BwdsInst = @opaque (j::Int) -> j
    return fwds_inst, bwds_inst
end

#
# Code execution
#

function rrule!!(::CoDual{typeof(_eval)}, fargs::Vararg{CoDual, N}) where {N}
    out, pb!! = rrule!!(fargs...)
    _eval_pb!!(dout, d_eval, dfargs...) = d_eval, pb!!(dout, dfargs...)...
    return out, _eval_pb!!
end

function rrule!!(_f::CoDual{<:DelayedInterpretedFunction{C, F}}, args::CoDual...) where {C, F}
    f = primal(_f)
    s = _typeof(map(primal, args))
    if is_primitive(C, s)
        return rrule!!(zero_codual(f.f), args...)
    else
        in_f = InterpretedFunction(f.ctx, s, f.interp)
        return build_rrule!!(in_f)(zero_codual(in_f), args...)
    end
end

tangent_type(::Type{<:InterpretedFunction}) = NoTangent
tangent_type(::Type{<:DelayedInterpretedFunction}) = NoTangent

function make_codual_arginfo(ai::ArgInfo{T, is_vararg}) where {T, is_vararg}
    codual_arg_slots = map(make_codual_stack, ai.arg_slots)
    return ArgInfo{_typeof(codual_arg_slots), is_vararg}(codual_arg_slots)
end

function load_rrule_args!(ai::ArgInfo{T, is_vararg}, args::Tuple) where {T, is_vararg}
    # There is a difference between the varargs that we recieve, and the varargs of the
    # original function. This section sorts that out.
    # For example if the original function is `f(x...)`, then the `argtypes` field of its
    # `IRCode` when calling e.g. `f(5.0)` will be `Tuple{typeof(f), Tuple{Float64}}`, where
    # the second tuple contains the vararg.
    # However, the `argtypes` field of the corresponding `InterpretedFunction` will
    # be `Tuple{<:InterpretedFunction, Tuple{typeof(f), Float64}}`.
    # Therefore, the `args` field of this function will be a `Tuple{typeof(f), Float64}`.
    # We must therefore transform it into a `Tuple` of type
    # `Tuple{typeof(f), Tuple{Float64}}` before attempting to load it into `ai.arg_slots`.
    if is_vararg
        num_args = length(ai.arg_slots) - 1 # once for first arg, once for vararg
        primals = map(primal, args)
        tangents = map(tangent, args)
        refined_primal_args = (primals[1:num_args]..., (primals[num_args+1:end]..., ))
        refined_tangent_args = (tangents[1:num_args]..., (tangents[num_args+1:end]..., ))
        refined_args = map(CoDual, refined_primal_args, refined_tangent_args)
    else
        refined_args = args
    end

    # Load the arguments into `ai.arg_slots`.
    return __push_args!(ai.arg_slots, refined_args)
end

@generated function __push_args!(arg_slots::Tuple, args::Tuple)
    Ts = args.parameters
    loaders = map(n -> :(push!(arg_slots[$n], args[$n])), eachindex(Ts))
    return Expr(:block, loaders..., :(return nothing))
end


struct InterpretedFunctionRRule{sig<:Tuple, Treturn, Targ_info<:ArgInfo}
    return_slot::SlotRef{Treturn}
    arg_info::Targ_info
    slots::Vector{CoDualStack}
    fwds_instructions::Vector{FwdsInst}
    bwds_instructions::Vector{BwdsInst}
    n_stack::Stack{Int}
    ir::IRCode
end

function _get_slot(x, in_f::InterpretedFunctionRRule)
    return _wrap_codual_slot(_get_slot(x, in_f.slots, in_f.arg_info, in_f.ir))
end

_wrap_codual_slot(x::CoDualSlot) = x
_wrap_codual_slot(x::ConstSlot{<:CoDual}) = x
_wrap_codual_slot(x::ConstSlot{P}) where {P} = ConstSlot{codual_type(P)}(zero_codual(x[]))

# Special handling is required for PhiNodes, because their semantics require that when
# more than one PhiNode appears at the start of a basic block, they are run simulataneously
# rather than in sequence. See the SSAIR docs for an explanation of why this is the case.
function make_phi_instructions!(
    in_f::InterpretedFunction, __rrule!!::InterpretedFunctionRRule
)
    ir = in_f.ir
    fwds_insts = __rrule!!.fwds_instructions
    bwds_insts = __rrule!!.bwds_instructions

    for (b, bb) in enumerate(ir.cfg.blocks)

        # Find any phi nodes at the start of the block.
        phi_node_inds = Int[]
        foreach(n -> (ir.stmts.inst[n] isa PhiNode) && push!(phi_node_inds, n), bb.stmts)
        isempty(phi_node_inds) && continue

        # Make a single instruction which runs all of the PhiNodes "simultaneously".
        # Specifically, this instruction runs all of the phi nodes, storing the results of
        # this into temporary storage, then writing from the temporary slots to the
        # final slots. This has the effect of ensuring that phi nodes that depend on other
        # phi nodes get the "old" values, not the new updated values. This was a
        # surprisingly hard bug to catch and resolve.
        nodes = [ir.stmts.inst[n] for n in phi_node_inds]
        n_first = first(phi_node_inds)
        is_blk_end = length(phi_node_inds) == length(bb.stmts)
        fwds_inst, bwds_inst = build_coinsts(nodes, in_f, __rrule!!, n_first, b, is_blk_end)
        fwds_insts[phi_node_inds[1]] = fwds_inst
        bwds_insts[phi_node_inds[1]] = bwds_inst

        # Create dummy instructions for the remainder of the nodes.
        for n in phi_node_inds[2:end]
            fwds_insts[n] = make_dummy_instruction(_standard_next_block(is_blk_end, b))
            bwds_insts[n] = make_dummy_instruction(n)
        end
    end
    return nothing
end

function build_rrule!!(in_f::InterpretedFunction{sig}) where {sig}

    return_slot = SlotRef{codual_type(eltype(in_f.return_slot))}()
    arg_info = make_codual_arginfo(in_f.arg_info)

    # Construct rrule!! for in_f.
    __rrule!! =  InterpretedFunctionRRule{sig, eltype(return_slot), _typeof(arg_info)}(
        return_slot,
        arg_info,
        map(make_codual_stack, in_f.slots), # SlotRefs
        Vector{FwdsInst}(undef, length(in_f.instructions)), # fwds_instructions
        Vector{BwdsInst}(undef, length(in_f.instructions)), # bwds_instructions
        Stack{Int}(),
        in_f.ir,
    )

    # Set PhiNodes.
    make_phi_instructions!(in_f, __rrule!!)

    return __rrule!!
end

struct InterpretedFunctionPb{Treturn_slot, Targ_info, Tbwds_f}
    j::Int
    bwds_instructions::Tbwds_f
    return_slot::Treturn_slot
    n_stack::Stack{Int}
    arg_info::Targ_info
end

function (in_f_rrule!!::InterpretedFunctionRRule{sig})(
    _in_f::CoDual{<:InterpretedFunction{sig}}, args::Vararg{CoDual, N}
) where {sig, N}

    # Load in variables.
    return_slot = in_f_rrule!!.return_slot
    arg_info = in_f_rrule!!.arg_info
    n_stack = in_f_rrule!!.n_stack

    # Initialise variables.
    load_rrule_args!(arg_info, args)
    in_f = primal(_in_f)
    prev_block = 0
    next_block = 0
    current_block = 1
    n = 1
    j = length(n_stack)

    # Run instructions until done.
    while next_block != -1
        push!(n_stack, n)
        if !isassigned(in_f_rrule!!.fwds_instructions, n)
            fwds, bwds = generate_coinstructions(in_f, in_f_rrule!!, n)
            in_f_rrule!!.fwds_instructions[n] = fwds
            in_f_rrule!!.bwds_instructions[n] = bwds
        end
        next_block = in_f_rrule!!.fwds_instructions[n](prev_block)
        if next_block == 0
            n += 1
        elseif next_block > 0
            n = in_f.bb_starts[next_block]
            prev_block = current_block
            current_block = next_block
            next_block = 0
        end
    end

    return_val = return_slot[]
    interpreted_function_pb!! = InterpretedFunctionPb(
        j, in_f_rrule!!.bwds_instructions, return_slot, n_stack, arg_info
    )
    return return_val, interpreted_function_pb!!
end

function flattened_rrule_args(ai::ArgInfo{T, is_vararg}) where {T, is_vararg}
    args = tuple_map(getindex, ai.arg_slots)
    !is_vararg && return args

    va_arg = args[end]
    return (args[1:end-1]..., map(CoDual, primal(va_arg), tangent(va_arg))...)
end

@noinline function load_tangents!(ai::ArgInfo{T, is_vararg}, dargs::Tuple) where {T, is_vararg}
    if is_vararg
        num_args = length(ai.arg_slots) - 1 # once for vararg
        refined_dargs = (dargs[1:num_args]..., (dargs[num_args+1:end]..., ))
    else
        refined_dargs = dargs
    end
    return __load_tangents!(ai.arg_slots, refined_dargs)
end

@generated function __load_tangents!(arg_slots::Tuple, dargs::Tuple)
    Ts = dargs.parameters
    loaders = map(n -> :(replace_tangent!(arg_slots[$n], dargs[$n])), eachindex(Ts))
    return Expr(:block, loaders..., :(return nothing))
end

function (if_pb!!::InterpretedFunctionPb)(dout, ::NoTangent, dargs::Vararg{Any, N}) where {N}

    # Update the output cotangent value to whatever is provided.
    replace_tangent!(if_pb!!.return_slot, dout)
    load_tangents!(if_pb!!.arg_info, dargs)

    # Run the instructions in reverse. Present assumes linear instruction ordering.
    n_stack = if_pb!!.n_stack
    bwds_instructions = if_pb!!.bwds_instructions
    while length(n_stack) > if_pb!!.j
        inst = bwds_instructions[pop!(n_stack)]
        inst(0)
    end

    # Return resulting tangents from slots.
    flat_arg_slots = flattened_rrule_args(if_pb!!.arg_info)
    return NoTangent(), tuple_map(tangent, flat_arg_slots)...
end

function generate_coinstructions(in_f, in_f_rrule!!, n)
    ir_inst = in_f.ir.stmts.inst[n]
    b = block_map(in_f.ir.cfg)[n]
    is_blk_end = n in in_f.bb_ends
    return build_coinsts(ir_inst, in_f, in_f_rrule!!, n, b, is_blk_end)
end

# Slow implementation, but useful for testing correctness.
function rrule!!(f_in::CoDual{<:InterpretedFunction}, args::CoDual...)
    return build_rrule!!(primal(f_in))(f_in, args...)
end
