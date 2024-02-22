# FwdsInsts have the same signature as Insts, but have different side-effects.
const FwdsInst = Core.OpaqueClosure{Tuple{Int}, Int}

# The backwards instructions don't actually need to return an Int, however, there is
# currently a performance bug in OpaqueClosures which means that an allocation is produced
# if a constant is returned. Consequently, we have to return something non-constant.
# See https://github.com/JuliaLang/julia/issues/52620 for info.
# By convention, any backwards instruction which "does nothing" just returns its input, and
# does no other work.
const BwdsInst = Core.OpaqueClosure{Tuple{Int}, Int}

const RuleSlot{V} = Union{SlotRef{V}, ConstSlot{V}} where {V<:Tuple{CoDual, Ref}}

__primal_type(::Type{<:Tuple{<:CoDual{P}, <:Any}}) where {P} = @isdefined(P) ? P : Any
function __primal_type(::Type{P}) where {P<:Tuple{<:CoDual, <:Any}}
    P isa Union && return Union{__primal_type(P.a), __primal_type(P.b)}
    return Any
end

primal_type(::AbstractSlot{P}) where {P} = __primal_type(P)

function rule_slot_type(::Type{P}) where {P}
    P isa Union && return Union{rule_slot_type(P.a), rule_slot_type(P.b)}
    return Tuple{codual_type(P), tangent_ref_type_ub(P)}
end

function make_rule_slot(::SlotRef{P}, ::Any) where {P}
    return SlotRef{rule_slot_type(P)}()
end
function make_rule_slot(x::ConstSlot{P}, ::Any) where {P}
    cd = uninit_codual(x[])
    stack = make_tangent_stack(P)
    push!(stack, tangent(cd))
    return ConstSlot((cd, top_ref(stack)))
end

make_tangent_stack(::Type{P}) where {P} = tangent_stack_type(P)()

make_tangent_ref_stack(::Type{P}) where {P} = Stack{P}()
make_tangent_ref_stack(::Type{NoTangentRef}) = NoTangentRefStack()

get_codual(x::RuleSlot) = x[][1]
get_tangent_stack(x::RuleSlot) = x[][2]

increment_ref!(x::Ref, t) = setindex!(x, increment!!(x[], t))

## ReturnNode
function build_coinsts(node::ReturnNode, _, _, _rrule!!, ::Int, ::Int, ::Bool)
    return build_coinsts(
        ReturnNode, _rrule!!.ret, _rrule!!.ret_tangent, _get_slot(node.val, _rrule!!),
    )
end
function build_coinsts(
    ::Type{ReturnNode}, ret::SlotRef{<:CoDual}, ret_tangent::SlotRef, val::RuleSlot
)
    tangent_ref_stack = Stack{tangent_ref_type_ub(primal_type(val))}()
    fwds_inst = @opaque function (p::Int)
        push!(tangent_ref_stack, get_tangent_stack(val))
        ret[] = get_codual(val)
        return -1
    end
    bwds_inst = @opaque function (j::Int)
        increment_ref!(pop!(tangent_ref_stack), ret_tangent[])
        return j
    end
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## GotoNode
build_coinsts(x::GotoNode, _, _, _, ::Int, ::Int, ::Bool) = build_coinsts(GotoNode, x.label)
function build_coinsts(::Type{GotoNode}, dest::Int)
    return build_inst(GotoNode, dest)::FwdsInst, (@opaque (j::Int) -> j)::BwdsInst
end

## GotoIfNot
function build_coinsts(x::GotoIfNot, _, _, _rrule!!, ::Int, b::Int, is_blk_end::Bool)
    return build_coinsts(GotoIfNot, x.dest, b + 1, _get_slot(x.cond, _rrule!!))
end
function build_coinsts(::Type{GotoIfNot}, dest::Int, next_blk::Int, cond::RuleSlot)
    fwds_inst = @opaque (p::Int) -> primal(get_codual(cond)) ? next_blk : dest
    bwds_inst = @opaque (j::Int) -> j
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## PhiNode
function build_coinsts(ir_insts::Vector{PhiNode}, _, _rrule!!, n_first::Int, b::Int, is_blk_end::Bool)
    nodes = (build_typed_phi_nodes(ir_insts, _rrule!!, n_first)..., )
    next_blk = _standard_next_block(is_blk_end, b)
    return build_coinsts(Vector{PhiNode}, nodes, next_blk)
end
function build_coinsts(
    ::Type{Vector{PhiNode}}, nodes::NTuple{N, TypedPhiNode}, next_blk::Int,
) where {N}
    # Check that we're operating on CoDuals.
    @assert all(x -> x.ret_slot isa RuleSlot, nodes)
    @assert all(x -> all(y -> isa(y, RuleSlot), x.values), nodes)

    # Construct instructions.
    fwds_inst = build_inst(Vector{PhiNode}, nodes, next_blk)
    bwds_inst = @opaque (j::Int) -> j
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## PiNode
function build_coinsts(x::PiNode, P, _, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
    val = _get_slot(x.val, _rrule!!)
    ret = _rrule!!.slots[n]
    return build_coinsts(PiNode, P, val, ret, _standard_next_block(is_blk_end, b))
end
function build_coinsts(
    ::Type{PiNode},
    ::Type{P},
    val::RuleSlot,
    ret::RuleSlot{<:Tuple{R, <:Any}},
    next_blk::Int,
) where {R, P}

    my_tangent_stack = make_tangent_stack(P)
    tangent_stack_stack = make_tangent_ref_stack(tangent_ref_type_ub(primal_type(val)))

    make_fwds(v) = R(primal(v), tangent(v))
    function fwds_run()
        v, tangent_stack = val[]
        push!(my_tangent_stack, tangent(v))
        push!(tangent_stack_stack, tangent_stack)
        ret[] = (make_fwds(v), top_ref(my_tangent_stack))
        return next_blk
    end
    fwds_inst = @opaque (p::Int) -> fwds_run()
    function bwds_run()
        increment_ref!(pop!(tangent_stack_stack), pop!(my_tangent_stack))
    end
    bwds_inst = @opaque (j::Int) -> (bwds_run(); return j)
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## GlobalRef
function build_coinsts(x::GlobalRef, P, _, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
    next_blk = _standard_next_block(is_blk_end, b)
    return build_coinsts(GlobalRef, P, _globalref_to_slot(x), _rrule!!.slots[n], next_blk)
end
function build_coinsts(
    ::Type{GlobalRef}, ::Type{P}, global_ref::AbstractSlot, out::RuleSlot, next_blk::Int
) where {P}
    my_tangent_stack = make_tangent_stack(P)
    fwds_inst = @opaque function (p::Int)
        v = uninit_codual(global_ref[])
        push!(my_tangent_stack, tangent(v))
        out[] = (v, top_ref(my_tangent_stack))
        return next_blk
    end
    bwds_inst = @opaque function (j::Int)
        pop!(my_tangent_stack)
        return j
    end
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

## QuoteNode and literals
function build_coinsts(node, _, _, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
    x = ConstSlot(zero_codual(node isa QuoteNode ? node.value : node))
    next_blk = _standard_next_block(is_blk_end, b)
    return build_coinsts(nothing, x, _rrule!!.slots[n], next_blk)
end
function build_coinsts(::Nothing, x::ConstSlot, out::RuleSlot, next_blk::Int)
    my_tangent_stack = make_tangent_stack(primal_type(out))
    push!(my_tangent_stack, tangent(x[]))
    fwds_inst = @opaque function (p::Int)
        out[] = (x[], top_ref(my_tangent_stack))
        return next_blk
    end
    bwds_inst = @opaque (j::Int) -> j
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
    codual_sig = Tuple{_typeof(deval), map(codual_type ∘ primal_type, arg_slots)...}
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
        F = T_pb!!.parameters[2]
        return Base.issingletontype(F) ? SingletonStack{F}() : Stack{F}()
    else
        return Stack{Any}()
    end
end

function build_coinsts(ir_inst::Expr, P, in_f, _rrule!!, n::Int, b::Int, is_blk_end::Bool)
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
        primal_sig = Tuple{map(arg -> eltype(_get_slot(arg, in_f)), (__args..., ))...}
        evaluator = get_evaluator(in_f.ctx, primal_sig, in_f.interp, is_invoke)
        __rrule!! = get_rrule!!_evaluator(evaluator)

        # Create stack for storing pullbacks.
        pb_stack = build_pb_stack(__rrule!!, evaluator, arg_slots)

        return build_coinsts(
            Val(:call), P, val_slot, arg_slots, evaluator, __rrule!!, pb_stack, next_blk
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

function build_coinsts(::Val{:boundscheck}, out::RuleSlot, next_blk::Int)
    @assert eltype(out) == Tuple{CoDual{Bool, NoTangent}, NoTangentRef}
    fwds_inst = @opaque function (p::Int)
        out[] = (zero_codual(true), NoTangentRef())
        return next_blk
    end
    bwds_inst = @opaque (j::Int) -> j
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

function build_coinsts(
    ::Val{:call},
    ::Type{P},
    out::RuleSlot,
    arg_slots::NTuple{N, RuleSlot} where {N},
    evaluator::Teval,
    __rrule!!::Trrule!!,
    pb_stack,
    next_blk::Int,
) where {P, Teval, Trrule!!}

    my_tangent_stack = make_tangent_stack(P)

    tangent_stack_stacks = map(arg_slots) do arg_slot
        make_tangent_ref_stack(tangent_ref_type_ub(primal_type(arg_slot)))
    end

    function fwds_pass()
        args = tuple_map(get_codual, arg_slots)
        map(tangent_stack_stacks, arg_slots) do tangent_stack_stack, arg
            push!(tangent_stack_stack, get_tangent_stack(arg))
        end
        _out, pb!! = __rrule!!(zero_codual(evaluator), args...)
        push!(my_tangent_stack, tangent(_out))
        push!(pb_stack, pb!!)
        out[] = (_out, top_ref(my_tangent_stack))
        return nothing
    end
    fwds_inst = @opaque function (p::Int)
        fwds_pass()
        return next_blk
    end

    function bwds_pass()
        pb!! = pop!(pb_stack)
        dout = pop!(my_tangent_stack)
        tangent_stacks = map(pop!, tangent_stack_stacks)
        dargs = tuple_map(set_immutable_to_zero ∘ getindex, tangent_stacks)
        new_dargs = pb!!(dout, NoTangent(), dargs...)
        map(increment_ref!, tangent_stacks, new_dargs[2:end])
        return nothing
    end
    bwds_inst = @opaque function (j::Int)
        bwds_pass()
        return j
    end
    # display(Base.code_ircode(fwds_pass, Tuple{}))
    # display(Base.code_ircode(bwds_pass, Tuple{}))
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
end

function build_coinsts(::Val{:skipped_expression}, next_blk::Int)
    return (@opaque (p::Int) -> next_blk), (@opaque (j::Int) -> j)
end

function build_coinsts(::Val{:throw_undef_if_not}, slot::AbstractSlot, next_blk::Int)
    fwds_inst = @opaque function (prev_blk::Int)
        !isassigned(slot) && throw(error("Boooo, not assigned"))
        return next_blk
    end
    bwds_inst = @opaque (j::Int) -> j
    return fwds_inst::FwdsInst, bwds_inst::BwdsInst
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
        return rrule!!(zero_codual(_eval), args...)
    else
        in_f = InterpretedFunction(f.ctx, s, f.interp)
        return build_rrule!!(in_f)(zero_codual(in_f), args...)
    end
end

tangent_type(::Type{<:InterpretedFunction}) = NoTangent
tangent_type(::Type{<:DelayedInterpretedFunction}) = NoTangent

function make_codual_arginfo(ai::ArgInfo{T, is_vararg}) where {T, is_vararg}
    arg_slots = map(Base.Fix2(make_rule_slot, nothing), ai.arg_slots)
    return ArgInfo{_typeof(arg_slots), is_vararg}(arg_slots)
end

function make_arg_tangent_stacks(argtypes::Vector{Any})
    return map(a -> tangent_stack_type(a)(), (map(_get_type, argtypes)...,))
end

function load_rrule_args!(
    ai::ArgInfo{T, is_vararg}, args::Tuple, arg_tangent_stacks::Tuple
) where {T, is_vararg}
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
    map(refined_args, arg_tangent_stacks) do arg, arg_tangent_stack
        push!(arg_tangent_stack, tangent(arg))
    end
    args = map((a, b) -> (a, top_ref(b)), refined_args, arg_tangent_stacks)
    return __load_args!(ai.arg_slots, args)
end

struct InterpretedFunctionRRule{
    sig<:Tuple, Tret, Tret_tangent, Targ_info<:ArgInfo, Targ_tangent_stacks
}
    ret::SlotRef{Tret}
    ret_tangent::SlotRef{Tret_tangent}
    arg_info::Targ_info
    arg_tangent_stacks::Targ_tangent_stacks
    slots::Vector{RuleSlot}
    fwds_instructions::Vector{FwdsInst}
    bwds_instructions::Vector{BwdsInst}
    n_stack::Stack{Int}
    ir::IRCode
end

function _get_slot(x, in_f::InterpretedFunctionRRule)
    return _wrap_rule_slot(_get_slot(x, in_f.slots, in_f.arg_info, in_f.ir))
end

_wrap_rule_slot(x::RuleSlot) = x
function _wrap_rule_slot(x::ConstSlot{<:CoDual})
    stack = make_tangent_stack(primal_type(x))
    push!(stack, tangent(x[]))
    return ConstSlot((x[], top_ref(stack)))
end
function _wrap_rule_slot(x::ConstSlot{P}) where {P}
    T = tangent_type(P)
    Tref = tangent_ref_type_ub(P)
    stack = make_tangent_stack(P)
    push!(stack, zero_tangent(x[]))
    return ConstSlot{Tuple{codual_type(P), Tref}}((zero_codual(x[]), top_ref(stack)))
end

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
    return_tangent_slot = SlotRef{tangent_type(eltype(in_f.return_slot))}()
    arg_info = make_codual_arginfo(in_f.arg_info)
    arg_tangent_stacks = make_arg_tangent_stacks(in_f.ir.argtypes)

    # Construct rrule!! for in_f.
    Tret = eltype(return_slot)
    Tret_tangent = eltype(return_tangent_slot)
    __rrule!! =  InterpretedFunctionRRule{
        sig, Tret, Tret_tangent, _typeof(arg_info), _typeof(arg_tangent_stacks)
    }(
        return_slot,
        return_tangent_slot,
        arg_info,
        arg_tangent_stacks,
        RuleSlot[
            make_rule_slot(primal_slot, inst) for 
                (primal_slot, inst) in zip(in_f.slots, in_f.ir.stmts.inst)
        ], # SlotRefs
        Vector{FwdsInst}(undef, length(in_f.instructions)), # fwds_instructions
        Vector{BwdsInst}(undef, length(in_f.instructions)), # bwds_instructions
        Stack{Int}(),
        in_f.ir,
    )

    # Set PhiNodes.
    make_phi_instructions!(in_f, __rrule!!)

    return __rrule!!
end

struct InterpretedFunctionPb{Tret_tangent<:SlotRef, Targ_info, Tbwds_f, V, Q}
    j::Int
    bwds_instructions::Tbwds_f
    ret_tangent::Tret_tangent
    n_stack::Stack{Int}
    arg_info::Targ_info
    arg_tangent_stacks::V
    arg_tangent_stack_refs::Q
end

function (in_f_rrule!!::InterpretedFunctionRRule{sig})(
    _in_f::CoDual{<:InterpretedFunction{sig}}, args::Vararg{CoDual, N}
) where {sig, N}

    # Load in variables.
    return_slot = in_f_rrule!!.ret
    arg_info = in_f_rrule!!.arg_info
    arg_tangent_stacks = in_f_rrule!!.arg_tangent_stacks
    n_stack = in_f_rrule!!.n_stack

    # Initialise variables.
    load_rrule_args!(arg_info, args, arg_tangent_stacks)
    in_f = primal(_in_f)
    prev_block = 0
    next_block = 0
    current_block = 1
    n = 1
    j = length(n_stack)

    # Get references to top of tangent stacks for use on reverse-pass.
    arg_tangent_stack_refs = map(top_ref, arg_tangent_stacks)

    # Run instructions until done.
    while next_block != -1
        if !isassigned(in_f_rrule!!.fwds_instructions, n)
            fwds, bwds = generate_coinstructions(in_f, in_f_rrule!!, n)
            in_f_rrule!!.fwds_instructions[n] = fwds
            in_f_rrule!!.bwds_instructions[n] = bwds
        end
        next_block = in_f_rrule!!.fwds_instructions[n](prev_block)
        push!(n_stack, n)
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
        j,
        in_f_rrule!!.bwds_instructions,
        in_f_rrule!!.ret_tangent,
        n_stack,
        arg_info,
        arg_tangent_stacks,
        arg_tangent_stack_refs,
    )
    return return_val, interpreted_function_pb!!
end

function (if_pb!!::InterpretedFunctionPb)(dout, ::NoTangent, dargs::Vararg{Any, N}) where {N}

    # Update the output cotangent value to whatever is provided.
    if_pb!!.ret_tangent[] = dout
    tangent_stack_refs = if_pb!!.arg_tangent_stack_refs # this can go when we refactor
    set_tangent_stacks!(tangent_stack_refs, dargs, if_pb!!.arg_info)

    # Run the instructions in reverse. Present assumes linear instruction ordering.
    n_stack = if_pb!!.n_stack
    bwds_instructions = if_pb!!.bwds_instructions
    while length(n_stack) > if_pb!!.j
        inst = bwds_instructions[pop!(n_stack)]
        inst(0)
    end

    # Return resulting tangents from slots.
    return NoTangent(), assemble_dout(if_pb!!.arg_tangent_stacks, if_pb!!.arg_info)...
end

function set_tangent_stacks!(tangent_stacks, dargs, ai::ArgInfo{<:Any, is_va}) where {is_va}
    refined_dargs = unflatten_vararg(Val(is_va), dargs, Val(length(ai.arg_slots) - 1))
    map(setindex!, tangent_stacks, refined_dargs)
end

function assemble_dout(tangent_stacks, ::ArgInfo{<:Any, is_va}) where {is_va}
    dargs = map(pop!, tangent_stacks)
    return is_va ? (dargs[1:end-1]..., dargs[end]...) : dargs
end

function generate_coinstructions(in_f, in_f_rrule!!, n)
    ir_inst = in_f.ir.stmts.inst[n]
    ir_type = _get_type(in_f.ir.stmts.type[n])
    b = block_map(in_f.ir.cfg)[n]
    is_blk_end = n in in_f.bb_ends
    return build_coinsts(ir_inst, ir_type, in_f, in_f_rrule!!, n, b, is_blk_end)
end

# Slow implementation, but useful for testing correctness.
function rrule!!(f_in::CoDual{<:InterpretedFunction}, args::CoDual...)
    return build_rrule!!(primal(f_in))(f_in, args...)
end
