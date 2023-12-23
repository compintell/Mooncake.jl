# FwdsInsts have the same signature as Insts, but have different side-effects.
const FwdsInst = Core.OpaqueClosure{Tuple{Int}, Int}

# The backwards instructions don't actually need to return an Int, however, there is
# currently a performance bug in OpaqueClosures which means that an allocation is produced
# if a constant is returned. Consequently, we have to return something non-constant.
# See https://github.com/JuliaLang/julia/issues/52620 for info.
# By convention, any backwards instruction which "does nothing" just returns its input, and
# does no other work.
const BwdsInst = Core.OpaqueClosure{Tuple{Int}, Int}

const CoDualSlot = AbstractSlot{<:CoDual}

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

# ## PhiNode

# function increment_predecessor_from_tmp!(x::TypedPhiNode{<:CoDualSlot}, prev_blk::Int)
#     map(node.edges, node.values) do edge, v
#         (edge == prev_blk) && isassigned(v) && increment_tangent!(v, x.tmp_slot)
#     end
#     return nothing
# end

# function increment_tmp_from_return!(x::TypedPhiNode{<:CoDualSlot})
#     isassigned(x.return_slot) && increment_tangent!(x.tmp_slot, x.return_slot)
#     return nothing
# end

# function build_coinsts(ir_insts::Vector{PhiNode}, _, _rrule!!, n_first::Int, b::Int, is_blk_end::Bool)

#     typed_phi_nodes = build_typed_phi_nodes(ir_insts, _rrule!!, n_first)
#     tmp_slots_stack = Vector{eltype(val_slot)}(undef, 0)
#     value_slots_stack = Vector{eltype(val_slot)}(undef, 0)
#     prev_block_stack = Vector{Int}(undef, 0)



#     # Construct operation to run the forwards-pass.
#     run_fwds_pass::FwdsInst = @opaque function (prev_blk::Int, current_blk::Int)
#         push!(prev_block_stack, prev_blk)

#         for n in eachindex(edges)
#             if edges[n] == prev_blk
#                 if isassigned(val_slot)
#                     push!(value_slot_stack, val_slot[])
#                 end
#                 val_slot[] = extract_arg(values[n])
#             end
#         end
#         return is_blk_end ? current_blk + 1 : 0
#     end

#     # Construct operation to run the reverse-pass.
#     run_rvs_pass::BwdsInst = @opaque function (j::Int)
#         prev_block = pop!(prev_block_stack)
#         for n in eachindex(edges)
#             if edges[n] == prev_block
#                 replace_tangent!(
#                     values[n], increment!!(tangent(values[n][]), tangent(val_slot[])),
#                 )
#                 if !isempty(value_slot_stack)
#                     val_slot[] = pop!(value_slot_stack)
#                 end
#             end
#         end
#         return j
#     end

#     return run_fwds_pass, run_rvs_pass
# end

# PiNode

# function build_coinstructions(ir_inst::PiNode, _, in_f_rrule!!, n::Int, is_blk_end::Bool)
#     input_ref = _get_slot(ir_inst.val, in_f_rrule!!)
#     val_ref = in_f_rrule!!.slots[n]
#     run_fwds_pass::FwdsInst = @opaque function(::Int, current_blk::Int)
#         val_ref[] = input_ref[]
#         return is_blk_end ? current_blk + 1 : 0
#     end
#     run_rvs_pass::BwdsInst = @opaque function(j::Int)
#         input_ref[] = val_ref[]
#         return j
#     end
#     return run_fwds_pass, run_rvs_pass
# end

#
# GlobalRef
#

#
# QuoteNode and literals
#

# function build_coinstructions(node::TypedGlobalRef, _, in_f_rrule!!, n, is_blk_end)
#     function __barrier(c::A, val_ref::B) where {A, B}
#         run_fwds_pass = @opaque function (a::Int, current_blk::Int)
#             val_ref[] = c
#             return is_blk_end ? current_blk + 1 : 0
#         end
#         run_rvs_pass = @opaque (j::Int) -> j
#         return run_fwds_pass, run_rvs_pass
#     end
#     return __barrier(uninit_codual(node[]), in_f_rrule!!.slots[n])
# end

#
# Expr
#

# function replace_tangent!(x::SlotRef{<:CoDual{Tx, Tdx}}, new_tangent::Tdx) where {Tx, Tdx}
#     x_val = x[]
#     x[] = CoDual(primal(x_val), new_tangent)
#     return nothing
# end

# function replace_tangent!(x::SlotRef{<:CoDual}, new_tangent)
#     x_val = x[]
#     x[] = CoDual(primal(x_val), new_tangent)
#     return nothing
# end

# # Handles the case where `x` is a constant, rather than a slot.
# replace_tangent!(x, new_tangent) = nothing

# function build_coinstructions(ir_inst::Expr, in_f, in_f_rrule!!, n, is_blk_end)
#     is_invoke = Meta.isexpr(ir_inst, :invoke)
#     if is_invoke || Meta.isexpr(ir_inst, :call)

#         # Extract args refs.
#         __args = is_invoke ? ir_inst.args[3:end] : ir_inst.args[2:end]
#         codual_arg_refs = map(arg -> _get_slot(arg, in_f_rrule!!), (__args..., ))

#         # Extract val ref.
#         codual_val_ref = in_f_rrule!!.slots[n]

#         # Extract function.
#         fn = is_invoke ? ir_inst.args[2] : ir_inst.args[1]
#         if fn isa Core.SSAValue || fn isa Core.Argument
#             fn = primal(_get_slot(fn, in_f_rrule!!)[])
#         end
#         if fn isa GlobalRef
#             fn = getglobal(fn.mod, fn.name)
#         end
#         if fn isa Core.IntrinsicFunction
#             fn = IntrinsicsWrappers.translate(Val(fn))
#         end

#         fn_sig = Tuple{map(eltype ∘ primal ∘ extract_codual, codual_arg_refs)...}
#         __rrule!! = rrule!!
#         if !is_primitive(in_f.ctx, fn_sig)
#             if all(Base.isconcretetype, fn_sig.parameters)
#                 fn = InterpretedFunction(in_f.ctx, fn_sig, in_f.interp)
#                 __rrule!! = build_rrule!!(fn)
#             else
#                 fn = DelayedInterpretedFunction{Core.Typeof(in_f.ctx), Core.Typeof(fn)}(
#                     in_f.ctx, fn, in_f.interp
#                 )
#             end
#         end

#         # Wrap f to make it rrule!!-friendly.
#         fn = uninit_codual(fn)

#         # Create stacks for storing intermediates.
#         codual_sig = Tuple{map(eltype ∘ extract_codual, codual_arg_refs)...}
#         output = Base.return_types(__rrule!!, codual_sig)
#         if length(output) == 0
#             throw(error("No return type inferred for __rrule!! with sig $codual_sig"))
#         elseif length(output) > 1
#             @warn "Too many output types inferred"
#             display(output)
#             println()
#             throw(error("> 1 return type inferred for __rrule!! with sig $codual_sig "))
#         end
#         T_pb!! = only(output)
#         if T_pb!! <: Tuple && T_pb!! !== Union{}
#             pb_stack = Vector{T_pb!!.parameters[2]}(undef, 0)
#         else
#             pb_stack = Vector{Any}(undef, 0)
#         end
#         sizehint!(pb_stack, 100)
#         old_vals = Vector{eltype(codual_val_ref)}(undef, 0)
#         sizehint!(old_vals, 100)

#         # Wrap any types in a data structure which prevents the introduction of type-
#         # instabilities.
#         lift(T::DataType) = TypeWrapper{T}()
#         lift(x) = x
#         codual_arg_refs = map(lift, codual_arg_refs)

#         function __barrier(
#             fn, codual_val_ref, __rrule!!, old_vals, pb_stack, is_blk_end, codual_arg_refs,
#         )

#             function ___fwds_pass(current_blk)
#                 if isassigned(codual_val_ref)
#                     push!(old_vals, codual_val_ref[])
#                 end
#                 out, pb!! = __rrule!!(fn, map(extract_codual, codual_arg_refs)...)
#                 codual_val_ref[] = out
#                 push!(pb_stack, pb!!)
#                 return is_blk_end ? current_blk + 1 : 0
#             end

#             # Construct operation to run the forwards-pass.
#             run_fwds_pass = @opaque function (a::Int, current_blk::Int)
#                 ___fwds_pass(current_blk)
#             end
#             if !(run_fwds_pass isa FwdsInst)
#                 @warn "Unable to compiled forwards pass -- running to generate the error."
#                 @show run_fwds_pass(5, 4)
#             end

#             # Construct operation to run the reverse-pass.
#             run_rvs_pass = @opaque function (j::Int)
#                 dout = tangent(codual_val_ref[])
#                 dargs = map(tangent, map(extract_codual, codual_arg_refs))
#                 _, new_dargs... = pop!(pb_stack)(dout, tangent(fn), dargs...)
#                 map(replace_tangent!, codual_arg_refs, new_dargs)
#                 if !isempty(old_vals)
#                     codual_val_ref[] = pop!(old_vals) # restore old state.
#                 end
#                 return j
#             end
#             if !(run_rvs_pass isa BwdsInst)
#                 @warn "Unable to compiled reverse pass -- running to generate the error."
#                 @show run_reverse_pass(5)
#             end

#             return run_fwds_pass, run_rvs_pass
#         end
#         return __barrier(
#             fn, codual_val_ref, __rrule!!, old_vals, pb_stack, is_blk_end, codual_arg_refs
#         )
#     elseif ir_inst isa Expr && ir_inst.head in [
#         :code_coverage_effect, :gc_preserve_begin, :gc_preserve_end, :loopinfo,
#         :leave, :pop_exception,
#     ]
#         run_fwds_pass = @opaque (a::Int, b::Int) -> 0
#         run_rvs_pass = @opaque (j::Int) -> j
#         return run_fwds_pass, run_rvs_pass
#     else
#         throw(error("Unrecognised expression $ir_inst"))
#     end
# end

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
    if isconcretetype(P)
        return SlotRef{CoDual{P, tangent_type(P)}}()
    else
        return SlotRef{CoDual}()
    end
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
