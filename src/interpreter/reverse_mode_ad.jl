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

# Operations on Slots involving CoDuals

function increment_tangent!(x::SlotRef{<:CoDual}, y::CoDualSlot)
    x_val = x[]
    x[] = CoDual(primal(x_val), increment!!(tangent(x_val), tangent(y[])))
    return nothing
end

increment_tangent!(x::ConstSlot{<:CoDual}, new_tangent) = nothing

## ReturnNode
function build_coinsts(node::ReturnNode, _, _rrule!!, ::Int, ::Int, ::Bool)
    return build_coinsts(ReturnNode, _rrule!!.return_slot, _get_slot(node.val, _rrule!!))
end
function build_coinsts(::Type{ReturnNode}, ret_slot::SlotRef{<:CoDual}, val::CoDualSlot)
    fwds_inst = build_inst(ReturnNode, ret_slot, val)
    bwds_inst = @opaque (j::Int) -> (ret_slot[] = val[]; return j)
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## GotoNode
build_coinsts(x::GotoNode, in_f, _, ::Int, ::Int, ::Bool) = build_coinsts(GotoNode, x.dest)
function build_coinsts(::Type{GotoNode}, dest::Int)
    return build_inst(GotoNode, dest)::FwdsInst, (@opaque (j::Int) -> j)::BwdsInst
end

## GotoIfNot
function build_coinsts(x::GotoIfNot, in_f, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
    return build_coinsts(ir_inst.dest, b + 1, _get_slot(ir_inst.cond, _rrule!!))
end
function build_coinsts(::Type{GotoIfNot}, dest::Int, next_blk::Int, cond::CoDualSlot)
    fwds_inst = @opaque (p::Int) -> primal(cond[]) ? next_blk : dest
    bwds_inst = @opaque (j::Int) -> j
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## PhiNode

function make_stacks(nodes::NTuple{N, TypedPhiNode}) where {N}
    tmp_stacks = map(n -> Vector{eltype(n.tmp_slot)}(undef, 0), nodes)
    ret_stacks = map(n -> Vector{eltype(n.ret_slot)}(undef, 0), nodes)
    prev_blks = Vector{Int}(undef, 0)
    return tmp_stacks, ret_stacks, prev_blks
end

function build_coinsts(ir_insts::Vector{PhiNode}, _, _rrule!!, n_first::Int, b::Int, is_blk_end::Bool)
    nodes = build_typed_phi_nodes(ir_insts, _rrule!!, n_first)
    next_blk = _standard_next_block(is_blk_end, b)
    return build_coinsts(Vector{PhiNode}, nodes, next_blk, make_stacks(nodes)...)
end

function increment_predecessor_from_tmp!(x::TypedPhiNode{<:CoDualSlot}, prev_blk::Int)
    map(x.edges, x.values) do edge, v
        (edge == prev_blk) && isassigned(v) && increment_tangent!(v, x.tmp_slot)
    end
    return nothing
end

function increment_tmp_from_return!(x::TypedPhiNode{<:CoDualSlot})
    isassigned(x.ret_slot) && increment_tangent!(x.tmp_slot, x.ret_slot)
    return nothing
end

function build_coinsts(
    ::Type{Vector{PhiNode}},
    nodes::NTuple{N, TypedPhiNode},
    next_blk::Int,
    tmp_stacks::NTuple{N, Vector},
    ret_stacks::NTuple{N, Vector},
    prev_stack::Vector{Int},
) where {N}

    # Check that we're operating on CoDuals.
    @assert all(x -> x.ret_slot isa CoDualSlot, nodes)
    @assert all(x -> all(y -> isa(y, CoDualSlot), x.values), nodes)

    # Pre-allocate a little bit of memory.
    sizehint!(prev_stack, 10)
    foreach(Base.Fix2(sizehint!, 10), tmp_stacks)
    foreach(Base.Fix2(sizehint!, 10), ret_stacks)

    # Construct instructions.
    fwds_inst = @opaque function (p::Int)
        push!(prev_stack, p)
        map((n, t) -> isassigned(n.tmp_slot) && push!(t, n.tmp_slot[]), nodes, tmp_stacks)
        map(Base.Fix2(store_tmp_value!, p), nodes)
        map((n, r) -> isassigned(n.ret_slot) && push!(r, n.ret_slot[]), nodes, ret_stacks)
        map(transfer_tmp_value!, nodes)
        return next_blk
    end
    bwds_inst = @opaque function (j::Int)
        p = pop!(prev_stack)
        map(increment_tmp_from_return!, nodes)
        map((n, r) -> (!isempty(r)) && (n.ret_slot[] = pop!(r)), nodes, ret_stacks)
        map(Base.Fix2(increment_predecessor_from_tmp!, p), nodes)
        map((n, t) -> (!isempty(t)) && (n.tmp_slot[] = pop!(t)), nodes, tmp_stacks)
        return j
    end
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## PiNode
function build_coinsts(x::PiNode, _, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
    val = _get_slot(x.val, _rrule!!)
    ret = _get_slot(n, _rrule!!)
    return build_coinsts(PiNode, val, ret, _standard_next_block(is_blk_end, b))
end
function build_coinsts(
    ::Type{PiNode},
    val::AbstractSlot{<:CoDual{V, TV}},
    ret::CoDualSlot{<:CoDual{R, TR}},
    next_blk::Int,
) where {V, TV, R, TR}
    make_fwds(v) = CoDual{R, TR}(primal(v), tangent(v))
    make_bwds(r) = CoDual{V, TV}(primal(r), tangent(r))
    fwds_inst = @opaque (p::Int) -> (ret[] = make_fwds(val[]); return next_blk)
    bwds_inst = @opaque (j::Int) -> (val[] = make_bwds(ret[]); return j)
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## GlobalRef
function build_coinsts(x::GlobalRef, _, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
    next_blk = _standard_next_block(is_blk_end, b)
    return build_coinsts(GlobalRef, _globalref_to_slot(x), _get_slot(n, _rrule!!), next_blk)
end
function build_coinsts(::Type{GlobalRef}, x::AbstractSlot, out::CoDualSlot, next_blk::Int)
    fwds_inst = @opaque (p::Int) -> (out[] = zero_codual(x[]); return next_blk)
    bwds_inst = @opaque (j::Int) -> j
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## QuoteNode and literals
function build_coinsts(node, _, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
    x = ConstSlot(zero_codual(node isa QuoteNode ? node.value : node))
    next_blk = _standard_next_block(is_blk_end, b)
    return build_coinsts(nothing, x, _get_slot(n, _rrule!!), next_blk)
end
function build_coinsts(::Nothing, x::ConstSlot{<:CoDual}, out::CoDualSlot, next_blk::Int)
    fwds_inst = @opaque (p::Int) -> (out[] = x[]; return next_blk)
    bwds_inst = @opaque (j::Int) -> j
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## Expr

function build_inst(ir_inst::Expr, in_f, n::Int, b::Int, is_blk_end::Bool)::IFInstruction
    @nospecialize in_f
    ir_inst = preprocess_ir(ir_inst, in_f)
    next_blk = _standard_next_block(is_blk_end, b)
    val_slot = in_f.slots[n]
    if Meta.isexpr(ir_inst, :invoke) || Meta.isexpr(ir_inst, :call)

        # Extract args refs.
        __args = Meta.isexpr(ir_inst, :invoke) ? ir_inst.args[2:end] : ir_inst.args
        arg_refs = map(arg -> _get_slot(arg, in_f), (__args..., ))

        ctx = in_f.ctx
        sig = Tuple{map(eltype, arg_refs)...}
        evaluator = get_evaluator(ctx, sig, __args, in_f.interp)
        return build_inst(Val(:call), arg_refs, evaluator, val_slot, next_blk)
    end
end

function get_rrule!!_evaluator(ctx::T, sig, _, interp) where {T}
    is_primitive(ctx, sig) && return rrule!!
    if all(Base.isconcretetype, sig.parameters)
        return build_rrule!!(InterpretedFunction(ctx, sig, interp))
    else
        return rrule!! # very slow path
    end
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

        # Construct signature, and determine how the rrule is to be computed.
        ctx = in_f.ctx
        primal_sig = Tuple{map(eltype ∘ primal, arg_refs)...}
        __rrule!! = get_rrule!!_evaluator(ctx, primal_sig, __args, in_f.interp)

        # Create stack for storing pullbacks.
        codual_sig = Tuple{map(eltype, codual_arg_refs)...}
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
            pb_stack = Vector{T_pb!!.parameters[2]}(undef, 0)
        else
            pb_stack = Vector{Any}(undef, 0)
        end
        sizehint!(pb_stack, 100)

        # Create stack for storing values.
        old_vals = Vector{eltype(val_slot)}(undef, 0)
        sizehint!(old_vals, 100)

        return build_coinsts(
            Val(:call), val_slot, arg_slots, __rrule!!, old_vals, pb_stack, next_blk,
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

function replace_tangent!(x::SlotRef{<:CoDual{Tx, Tdx}}, new_tangent::Tdx) where {Tx, Tdx}
    x_val = x[]
    x[] = CoDual(primal(x_val), new_tangent)
    return nothing
end

function replace_tangent!(x::SlotRef{<:CoDual}, new_tangent)
    x_val = x[]
    x[] = CoDual(primal(x_val), new_tangent)
    return nothing
end

function build_coinsts(
    ::Val{:call},
    out::CoDualSlot,
    arg_slots::NTuple{N, CoDualSlot} where {N},
    __rrule!!::Trrule!!,
    old_vals::Vector,
    pb_stack::Vector,
    next_blk::Int,
) where {Trrule!!}
    fwds_inst = @opaque function (p::Int)
        isassigned(codual_val_ref) && push!(old_vals, out[])
        _out, pb!! = __rrule!!(tuple_map(getindex, arg_slots)...)
        out[] = _out
        push!(pb_stack, pb!!)
        return next_blk
    end
    bwds_inst = @opaque function (j::Int)
        dout = tangent(out[])
        dargs = tuple_map(tangent, tuple_map(extract_codual, arg_slots))
        new_dargs = pop!(pb_stack)(dout, dargs...)
        tuple_map(replace_tangent!, arg_slots, new_dargs)
        if !isempty(old_vals)
            out[] = pop!(old_vals) # restore old state.
        end
        return j
    end
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

function build_coinsts(::Val{:skipped_expression}, next_blk::Int)
    return (@opaque (p::Int) -> next_blk), (@opaque (j::Int) -> j)
end

#
# Code execution
#

function rrule!!(_f::CoDual{<:DelayedInterpretedFunction{C, F}}, args::CoDual...) where {C, F}
    f = primal(_f)
    s = Tuple{F, map(Core.Typeof ∘ primal, args)...}
    if is_primitive(f.ctx, s)
        return rrule!!(zero_codual(f.f), args...)
    else
        in_f = InterpretedFunction(f.ctx, s, f.interp)
        return build_rrule!!(in_f)(zero_codual(in_f), args...)
    end
end

tangent_type(::Type{<:InterpretedFunction}) = NoTangent
tangent_type(::Type{<:DelayedInterpretedFunction}) = NoTangent

# Pre-allocate for AD-related instructions and quantities.
function make_codual_slot(::SlotRef{P}) where {P}
    return isconcretetype(P) ? SlotRef{CoDual{P, tangent_type(P)}}() : SlotRef{CoDual}()
end

function make_codual_arginfo(ai::ArgInfo{T, is_vararg}) where {T, is_vararg}
    codual_arg_slots = map(make_codual_slot, ai.arg_slots)
    return ArgInfo{Core.Typeof(codual_arg_slots), is_vararg}(codual_arg_slots)
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
        num_args = length(ai.arg_slots) - 1 - 1 # once for first arg, once for vararg
        primals = map(primal, args)
        tangents = map(tangent, args)
        refined_primal_args = (primals[1:num_args]..., (primals[num_args+1:end]..., ))
        refined_tangent_args = (tangents[1:num_args]..., (tangents[num_args+1:end]..., ))
        refined_args = map(CoDual, refined_primal_args, refined_tangent_args)
    else
        refined_args = args
    end

    # Load the arguments into `ai.arg_slots`.
    return __load_args!(ai.arg_slots, refined_args)
end

function flattened_rrule_args(ai::ArgInfo{T, is_vararg}) where {T, is_vararg}
    args = map(getindex, ai.arg_slots[2:end])
    !is_vararg && return args

    va_arg = args[end]
    return (args[1:end-1]..., map(CoDual, primal(va_arg), tangent(va_arg))...)
end

struct InterpretedFunctionRRule{sig<:Tuple, Treturn, Targ_info<:ArgInfo}
    return_slot::SlotRef{Treturn}
    arg_info::Targ_info
    slots::Vector{SlotRef}
    fwds_instructions::Vector{FwdsInst}
    bwds_instructions::Vector{BwdsInst}
    n_stack::Vector{Int}
end

_get_slot(x, in_f::InterpretedFunctionRRule) = _get_slot(x, in_f.slots, in_f.arg_info)

function build_rrule!!(in_f::InterpretedFunction{sig}) where {sig}
    return_slot = make_codual_slot(in_f.return_slot)
    arg_info = make_codual_arginfo(in_f.arg_info)
    n_stack = Vector{Int}(undef, 1)
    sizehint!(n_stack, 100)
    return InterpretedFunctionRRule{sig, eltype(return_slot), Core.Typeof(arg_info)}(
        return_slot,
        arg_info,
        map(make_codual_slot, in_f.slots), # SlotRefs
        Vector{FwdsInst}(undef, length(in_f.instructions)), # fwds_instructions
        Vector{BwdsInst}(undef, length(in_f.instructions)), # bwds_instructions
        n_stack,
    )
end

struct InterpretedFunctionPb{Treturn_slot, Targ_info, Tbwds_f}
    j::Int
    bwds_instructions::Tbwds_f
    return_slot::Treturn_slot
    n_stack::Vector{Int}
    arg_info::Targ_info
end

function (in_f_rrule!!::InterpretedFunctionRRule{sig})(
    _in_f::CoDual{<:InterpretedFunction{sig}}, args::Vararg{CoDual, N}
) where {sig, N}

    # Load in variables.
    return_slot = in_f_rrule!!.return_slot
    arg_info = in_f_rrule!!.arg_info
    slots = in_f_rrule!!.slots
    n_stack = in_f_rrule!!.n_stack

    # Initialise variables.
    load_rrule_args!(arg_info, args)
    in_f = primal(_in_f)
    prev_block = 0
    next_block = 0
    current_block = 1
    n = 1
    j = 0

    # Run instructions until done.
    while next_block != -1
        j += 1
        if length(n_stack) >= j
            n_stack[j] = n
        else
            push!(n_stack, n)
        end

        if !isassigned(in_f.instructions, n) 
            in_f.instructions[n] = build_inst(in_f, n)
        end
        if !isassigned(in_f_rrule!!.fwds_instructions, n)
            fwds, bwds = generate_instructions(in_f, in_f_rrule!!, n)
            in_f_rrule!!.fwds_instructions[n] = fwds
            in_f_rrule!!.bwds_instructions[n] = bwds
        end
        next_block = in_f_rrule!!.fwds_instructions[n](prev_block, current_block)
        if next_block == 0
            n += 1
        elseif next_block > 0
            n = in_f.bb_starts[next_block]
            prev_block = current_block
            current_block = next_block
            next_block = 0
        end
    end

    interpreted_function_pb!! = InterpretedFunctionPb(
        j, in_f_rrule!!.bwds_instructions, return_slot, n_stack, arg_info
    )
    return return_slot[], interpreted_function_pb!!
end

function (if_pb!!::InterpretedFunctionPb)(dout, ::NoTangent, dargs::Vararg{Any, N}) where {N}

    # Update the output cotangent value to whatever is provided.
    replace_tangent!(if_pb!!.return_slot, dout)

    # Run the instructions in reverse. Present assumes linear instruction ordering.
    n_stack = if_pb!!.n_stack
    bwds_instructions = if_pb!!.bwds_instructions
    for i in reverse(1:if_pb!!.j)
        inst = bwds_instructions[n_stack[i]]
        inst(i)
    end

    # Increment and return.
    flat_arg_slots = flattened_rrule_args(if_pb!!.arg_info)
    new_dargs = map(dargs, flat_arg_slots[1:end]) do darg, arg_slot
        return increment!!(darg, tangent(arg_slot))
    end
    return NoTangent(), new_dargs...
end

const __Tinst = Tuple{FwdsInst, BwdsInst}

function generate_instructions(in_f, in_f_rrule!!, n)::__Tinst
    ir_inst = in_f.ir.stmts.inst[n]
    is_blk_end = n in in_f.bb_ends
    return build_coinstructions(ir_inst, in_f, in_f_rrule!!, n, is_blk_end)
end

# Slow implementation, but useful for testing correctness.
function rrule!!(f_in::CoDual{<:InterpretedFunction}, args::CoDual...)
    return build_rrule!!(primal(f_in))(f_in, args...)
end
