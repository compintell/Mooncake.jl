"""
    struct MinimalCtx end

Functions should only be primitives in this context if not making them so would cause AD to
fail. In particular, do not add primitives to this context if you are writing them for
performance only.
"""
struct MinimalCtx end

is_primitive(::MinimalCtx, ::Any) = false

"""
    @is_primitive context_type signature

Creates a method of `is_primitive` which always returns `true` for the context_type and
`signature` provided. For example
```julia
@is_primitive MinimalCtx Tuple{typeof(foo), Float64}
```
is equivalent to
```julia
is_primitive(::MinimalCtx, ::Type{<:Tuple{typeof(foo), Float64}}) = true
```

You should implemented more complicated method of `is_primitive` in the usual way.
"""
macro is_primitive(Tctx, sig)
    return :(Taped.is_primitive(::$Tctx, ::Type{<:$sig}) = true)
end

# @is_primitive MinimalCtx Tuple{Any}
@is_primitive MinimalCtx Tuple{typeof(rebind), Any}

"""
    struct DefaultCtx end

Context for all usually used AD primitives. Anything which is a primitive in a MinimalCtx is
a primitive in the DefaultCtx automatically. If you are adding a rule for the sake of
performance, it should be a primitive in the DefaultCtx, but not the MinimalCtx.
"""
struct DefaultCtx end

is_primitive(::DefaultCtx, sig) = is_primitive(MinimalCtx(), sig)

# Largely copied + extended from https://github.com/JuliaLang/julia/blob/2fe4190b3d26b4eee52b2b1b1054ddd6e38a941e/test/compiler/newinterp.jl#L11

struct TICache
    dict::IdDict{Core.MethodInstance, Core.CodeInstance}
end

TICache() = TICache(IdDict{Core.MethodInstance, Core.CodeInstance}())

struct TapedInterpreter{C} <: CC.AbstractInterpreter
    ctx::C
    meta # additional information
    world::UInt
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
    inf_cache::Vector{CC.InferenceResult}
    code_cache::TICache
    in_f_cache::Dict{Any, Any}
    function TapedInterpreter(
        ctx::C=DefaultCtx();
        meta=nothing,
        world::UInt=Base.get_world_counter(),
        inf_params::CC.InferenceParams=CC.InferenceParams(),
        opt_params::CC.OptimizationParams=CC.OptimizationParams(),
        inf_cache::Vector{CC.InferenceResult}=CC.InferenceResult[], 
        code_cache::TICache=TICache(),
        in_f_cache::Dict{Any, Any}=Dict(),
    ) where {C}
        return new{C}(
            ctx, meta, world, inf_params, opt_params, inf_cache, code_cache, in_f_cache
        )
    end
end

const TInterp = TapedInterpreter

CC.InferenceParams(interp::TInterp) = interp.inf_params
CC.OptimizationParams(interp::TInterp) = interp.opt_params
CC.get_world_counter(interp::TInterp) = interp.world
CC.get_inference_cache(interp::TInterp) = interp.inf_cache
function CC.code_cache(interp::TInterp)
    return CC.WorldView(interp.code_cache, CC.WorldRange(interp.world))
end
function CC.get(wvc::CC.WorldView{TICache}, mi::Core.MethodInstance, default)
    return get(wvc.cache.dict, mi, default)
end
function CC.getindex(wvc::CC.WorldView{TICache}, mi::Core.MethodInstance)
    return getindex(wvc.cache.dict, mi)
end
CC.haskey(wvc::CC.WorldView{TICache}, mi::Core.MethodInstance) = haskey(wvc.cache.dict, mi)
function CC.setindex!(
    wvc::CC.WorldView{TICache}, ci::Core.CodeInstance, mi::Core.MethodInstance
)
    return setindex!(wvc.cache.dict, ci, mi)
end

_type(x) = x
_type(x::CC.Const) = Core.Typeof(x.val)
_type(x::CC.PartialStruct) = x.typ
_type(x::CC.Conditional) = Union{x.thentype, x.elsetype}

function CC.inlining_policy(
    interp::TapedInterpreter{C},
    @nospecialize(src),
    @nospecialize(info::CC.CallInfo),
    stmt_flag::UInt8,
    mi::Core.MethodInstance,
    argtypes::Vector{Any},
) where {C}

    # Do not inline away primitives.
    argtype_tuple = Tuple{map(_type, argtypes)...}
    is_primitive(interp.ctx, argtype_tuple) && return nothing

    # If not a primitive, AD doesn't care about it. Use the usual inlining strategy.
    return @invoke CC.inlining_policy(
        interp::CC.AbstractInterpreter,
        src::Any,
        info::CC.CallInfo,
        stmt_flag::UInt8,
        mi::Core.MethodInstance,
        argtypes::Vector{Any},
    )
end

#
# Special types to represent data in an IRCode and a InterpretedFunction.
#

abstract type AbstractSlot{T} end

mutable struct SlotRef{T} <: AbstractSlot{T}
    x::T
    SlotRef{T}() where {T} = new{T}()
    SlotRef(x::T) where {T} = new{T}(x)
    SlotRef{T}(x) where {T} = new{T}(x)
end

Base.getindex(x::SlotRef) = getfield(x, :x)
function Base.setindex!(x::SlotRef, val)
    setfield!(x, :x, val)
    return x.x
end
Base.isassigned(x::SlotRef) = isdefined(x, :x)
Base.eltype(::SlotRef{T}) where {T} = T

struct ConstSlot{T} <: AbstractSlot{T}
    x::T
end

Base.getindex(x::ConstSlot) = getfield(x, :x)
Base.setindex!(x::ConstSlot, val) = nothing
Base.isassigned(x::ConstSlot) = true
Base.eltype(x::ConstSlot{T}) where {T} = T

#
# Utility functionality used through instruction construction.
#

const IFInstruction = Core.OpaqueClosure{Tuple{Int}, Int}
const FwdsIFInstruction = Core.OpaqueClosure{Tuple{Int}, Int}
const BwdsIFInstruction = Core.OpaqueClosure{Tuple{Int}, Int}
const JuliaSlot = Union{Argument, SSAValue}

# Standard handling for next-block returns for non control flow related instructions.
_standard_next_block(is_blk_end::Bool, current_blk::Int) = is_blk_end ? current_blk + 1 : 0

__empty_function(::Int) = 0

const default_ir = Base.code_ircode_by_type(Tuple{typeof(__empty_function), Int})[1][1]

# Various bits of functionality that we need acces to in order to work with IRCode.
Base.iterate(compact::CC.IncrementalCompact, state) = CC.iterate(compact, state)
Base.iterate(compact::CC.IncrementalCompact) = CC.iterate(compact)
Base.getindex(c::CC.IncrementalCompact, args...) = CC.getindex(c, args...)
Base.setindex!(c::CC.IncrementalCompact, args...) = CC.setindex!(c, args...)
Base.setindex!(i::CC.Instruction, args...) = CC.setindex!(i, args...)
Base.getindex(c::IRCode, args...) = CC.getindex(c, args...)
Base.setindex!(c::IRCode, args...) = CC.setindex!(c, args...)

new_inst(expr) = CC.NewInstruction(expr, Any, CC.NoCallInfo(), Int32(1), CC.IR_FLAG_REFINED)

#
# ReturnNode
#

preprocess_ir(st::ReturnNode, in_f) = ReturnNode(preprocess_ir(st.val, in_f))

# If the ReturnNode references an Argument or SlotRef, load it into the `ret_slot`.
# Otherwise, interpolate whatever we get (e.g. a `GlobalRef`) into the instruction.
function build_inst(inst::ReturnNode, in_f, ::Int, ::Int, ::Bool)
    if inst.val isa JuliaSlot
        return build_inst(ReturnNode, _get_input(inst.val, in_f), in_f.return_slot)
    else
        throw(error("Not implemented this branch yet"))
    end
end

function build_inst(::Type{ReturnNode}, ret_slot::SlotRef, val_slot::AbstractSlot)
    return @opaque function (prev_block::Int)
        ret_slot[] = val_slot[]
        return -1
    end
end

function build_inst(::Type{ReturnNode}, ret_slot::SlotRef, val)
    ir = CC.copy(default_ir)

    # Get ret_slot from environment, then set its value.
    CC.insert_node!(ir, 1, new_inst(Expr(:call, :getfield, Argument(1), 1)), false)
    CC.insert_node!(ir, 1, new_inst(Expr(:call, :setfield!, SSAValue(1), 1, val)), false)
    ir = Core.Compiler.compact!(ir)

    # Always return -1.
    ir[SSAValue(3)][:inst] = ReturnNode(-1)

    # Tidy up the IR and return.
    Core.Compiler.verify_ir(ir)
    return Core.OpaqueClosure(ir, ret_slot; do_compile=true)
end

# function build_coinstructions(ir_inst::ReturnNode, in_f, in_f_rrule!!, n, is_blk_end)
#     function __barrier(return_slot::A, slot_to_return::B) where {A, B}
#         # Construct operation to run the forwards-pass.
#         run_fwds_pass = @opaque function (a::Int, b::Int)
#             return_slot[] = extract_codual(slot_to_return)
#             return -1
#         end
#         if !(run_fwds_pass isa FwdsIFInstruction)
#             run_fwds_pass(5, 4)
#             display(CC.code_typed_opaque_closure(run_fwds_pass)[1][1])
#             println()
#         end

#         # Construct operation to run the reverse-pass.
#         run_rvs_pass = if slot_to_return isa SlotRef
#             @opaque  function (j::Int)
#                 setfield!(slot_to_return, :x, getfield(return_slot, :x))
#                 return j
#             end
#         else
#             @opaque (j::Int) -> j
#         end
#         if !(run_rvs_pass isa BwdsIFInstruction)
#             run_rvs_pass(4)
#             display(CC.code_typed_opaque_closure(run_fwds_pass)[1][1])
#             println()
#         end
#         return run_fwds_pass, run_rvs_pass
#     end
#     return __barrier(in_f_rrule!!.return_slot, _get_input(ir_inst.val, in_f_rrule!!))
# end

#
# GotoNode
#

preprocess_ir(st::GotoNode, _) = st

build_inst(inst::GotoNode, _, _, _, _) = build_inst(GotoNode, inst.label)

build_inst(::Type{GotoNode}, label::Int) = @opaque (p::Int) -> label

# function build_coinstructions(ir_inst::GotoNode, in_f, in_f_rrule!!, n, is_blk_end)
#     dest = ir_inst.label
#     run_fwds_pass::FwdsIFInstruction = @opaque (a::Int, b::Int) -> dest
#     run_rvs_pass::BwdsIFInstruction = @opaque (j::Int) -> j
#     return run_fwds_pass, run_rvs_pass
# end

#
# GotoIfNot
#

preprocess_ir(st::GotoIfNot, in_f) = GotoIfNot(preprocess_ir(st.cond, in_f), st.dest)

build_inst(x::GotoIfNot, in_f, n, b, _) = build_inst(GotoIfNot, x.cond, b + 1, x.dest)

function build_inst(::Type{GotoIfNot}, cond::AbstractSlot, next_blk::Int, dest::Int)
    return @opaque (p::Int) -> cond[] ? next_blk : dest
end

# function build_coinstructions(ir_inst::GotoIfNot, in_f, in_f_rrule!!, n, is_blk_end)
#     cond_slot = _get_input(ir_inst.cond, in_f_rrule!!)
#     dest = ir_inst.dest
#     run_fwds_pass::FwdsIFInstruction = @opaque function (a::Int, current_block::Int)
#         return primal(extract_codual(cond_slot)) ? current_block + 1 : dest
#     end
#     run_rvs_pass::BwdsIFInstruction = @opaque (j::Int) -> j
#     return run_fwds_pass, run_rvs_pass
# end

# #
# # PhiNode
# #

# # We can always safely assume that all `values` elements are SlotRefs.
# struct PhiNodeInst{Tedges<:Tuple, Tvalues<:Tuple, Tval_slot<:AbstractSlot}
#     edges::Tedges
#     values::Tvalues
#     val_slot::Tval_slot
#     next_blk::Int
#     node::PhiNode
#     line::Int
# end

# _isassigned(x::AbstractSlot) = isassigned(x)
# _isassigned(x) = true

# function (inst::PhiNodeInst)(prev_blk::Int)
#     for n in eachindex(inst.edges)
#         if inst.edges[n] == prev_blk && _isassigned(inst.values[n])
#             new_val = extract_arg(inst.values[n])
#             if new_val isa eltype(inst.val_slot)
#                 inst.val_slot[] = new_val
#             end
#         end
#     end
#     return inst.next_blk
# end

# function preprocess_ir(st::PhiNode, in_f)
#     new_vals = Vector{Any}(undef, length(st.values))
#     for n in eachindex(new_vals)
#         if isassigned(st.values, n)
#             new_vals[n] = preprocess_ir(st.values[n], in_f)
#         end
#     end
#     return PhiNode(st.edges, new_vals)
# end

# struct UndefinedReference end

# create_slot(x) = SlotRef(x)
# create_slot(x::Union{Literal, TypedGlobalRef, AbstractSlot}) = x

# function build_inst(ir_inst::PhiNode, in_f, n::Int, b::Int, is_blk_end::Bool)
#     edges = map(Int, (ir_inst.edges..., ))
#     values_vec_init = map(eachindex(ir_inst.values)) do j
#         if isassigned(ir_inst.values, j)
#             return _get_input(ir_inst.values[j], in_f)
#         else
#             return UndefinedReference()
#         end
#     end
#     T = eltype(values_vec_init)
#     values_vec = map(eachindex(values_vec_init)) do n
#         x = values_vec_init[n]
#         return x isa UndefinedReference ? SlotRef{T}() : create_slot(x)
#     end
#     values = (values_vec..., )
#     val_slot = in_f.slots[n]
#     next_blk = _standard_next_block(is_blk_end, b)
#     return PhiNodeInst(edges, values, val_slot, next_blk, ir_inst, n)
# end

# function build_coinstructions(ir_inst::PhiNode, _, in_f_rrule!!, n, is_blk_end)

#     # Extract relevant values.
#     edges = map(Int, (ir_inst.edges..., ))
#     values_vec = map(eachindex(ir_inst.values)) do j
#         if isassigned(ir_inst.values, j)
#             return _get_input(ir_inst.values[j], in_f_rrule!!)
#         else
#             return UndefinedReference()
#         end
#     end
#     values = map(x -> x isa SlotRef ? x : SlotRef(zero_codual(x)), (values_vec..., ))
#     val_slot = in_f_rrule!!.slots[n]

#     # Create a value slot stack.
#     value_slot_stack = Vector{eltype(val_slot)}(undef, 0)
#     prev_block_stack = Vector{Int}(undef, 0)

#     # Construct operation to run the forwards-pass.
#     run_fwds_pass::FwdsIFInstruction = @opaque function (prev_blk::Int, current_blk::Int)
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
#     run_rvs_pass::BwdsIFInstruction = @opaque function (j::Int)
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

# #
# # PiNode
# #

# struct PiNodeInst{Tinput_ref<:AbstractSlot, Tval_ref<:AbstractSlot}
#     input_ref::Tinput_ref
#     val_ref::Tval_ref
#     next_blk::Int
# end

# function (inst::PiNodeInst)(::Int)
#     inst.val_ref[] = inst.input_ref[]
#     return inst.next_blk
# end

# preprocess_ir(st::PiNode, in_f) = PiNode(preprocess_ir(st.val, in_f), st.typ)

# function build_inst(ir_inst::PiNode, in_f, n, b, is_blk_end)
#     next_blk = _standard_next_block(is_blk_end, b)
#     return PiNodeInst(_get_input(ir_inst.val, in_f), in_f.slots[n], next_blk)
# end

# function build_coinstructions(ir_inst::PiNode, _, in_f_rrule!!, n::Int, is_blk_end::Bool)
#     input_ref = _get_input(ir_inst.val, in_f_rrule!!)
#     val_ref = in_f_rrule!!.slots[n]
#     run_fwds_pass::FwdsIFInstruction = @opaque function(::Int, current_blk::Int)
#         val_ref[] = input_ref[]
#         return is_blk_end ? current_blk + 1 : 0
#     end
#     run_rvs_pass::BwdsIFInstruction = @opaque function(j::Int)
#         input_ref[] = val_ref[]
#         return j
#     end
#     return run_fwds_pass, run_rvs_pass
# end

# #
# # LiteralInst
# #

# struct LiteralInst{T, Tval_ref<:AbstractSlot}
#     val::Literal{T}
#     val_ref::Tval_ref
#     next_blk::Int
# end

# preprocess_ir(x::Literal, _) = x

# is_literal(::Union{Bool, Float16, Float32, Float64, Int, Nothing, DataType, Symbol}) = true
# is_literal(::Union{Val, UInt64, Char, UInt8, UInt16, UInt32, Function, Tuple, Type}) = true
# is_literal(::Union{UnionAll, Int8, Int16, Int32, Int128, UInt64, UInt128}) = true
# is_literal(_) = false

# function (inst::LiteralInst)(::Int)
#     inst.val_ref[] = inst.val[]
#     return inst.next_blk
# end

# function build_inst(val::Literal, in_f, n, b, is_blk_end)
#     return LiteralInst(val, in_f.slots[n], _standard_next_block(is_blk_end, b))
# end

# function build_coinstructions(val::Literal, _, in_f_rrule!!, n, is_blk_end)
#     function __barrier(val, val_ref, is_blk_end)
#         run_fwds_pass::FwdsIFInstruction = @opaque function(a::Int, current_block::Int)
#             val_ref[] = zero_codual(val.x)
#             return is_blk_end ? current_block + 1 : 0
#         end
#         run_rvs_pass::BwdsIFInstruction = @opaque (j::Int) -> j
#         return run_fwds_pass, run_rvs_pass
#     end
#     return __barrier(val, in_f_rrule!!.slots[n], is_blk_end)
# end

# #
# # GlobalRef
# #

# struct GlobalRefInst{Tc, Tval_ref<:AbstractSlot}
#     c::Tc
#     val_ref::Tval_ref
#     next_blk::Int
# end

# function (inst::GlobalRefInst)(::Int)
#     inst.val_ref[] = inst.c
#     return inst.next_blk
# end

# preprocess_ir(st::GlobalRef, _) = TypedGlobalRef(st)

# function build_inst(node::TypedGlobalRef, in_f, n::Int, b::Int, is_blk_end::Bool)
#     return GlobalRefInst(node[], in_f.slots[n], _standard_next_block(is_blk_end, b))
# end

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

# #
# # Expr -- this is a big one
# #

# struct CallInst{Targs<:NTuple{N, SlotRefOrLiteral} where {N}, T, Tval_ref<:AbstractSlot}
#     args::Targs
#     evaluator::T
#     val_ref::Tval_ref
#     next_blk::Int
#     ir::Expr
#     line::Int
# end

# function (inst::CallInst{sig, T})(::Int) where {sig, T}
#     inst.val_ref[] = inst.evaluator(map(extract_arg, inst.args)...)
#     return inst.next_blk
# end

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

# struct SkippedExpressionInst
#     s::Symbol
# end

# (::SkippedExpressionInst)(::Int) = 0

# struct ThrowUndefIfNot{Tval}
#     sym::Symbol
#     val::Tval
#     next_blk::Int
# end

# function (inst::ThrowUndefIfNot)(::Int)
#     if !_isassigned(inst.val)
#         throw(error("Boooo, not assigned"))
#     end
#     return inst.next_blk
# end

# _extract(x::Symbol) = x
# _extract(x::QuoteNode) = x.value

# function _preprocess_expr_arg(ex::Expr, sptypes)
#     if Meta.isexpr(ex, :boundscheck)
#         return Literal(true)
#     elseif Meta.isexpr(ex, :static_parameter)
#         ex = sptypes[ex.args[1]]
#         if ex isa CC.VarState
#             ex = ex.typ
#         end
#     else
#         return ex
#     end
# end

# _lift_intrinsic(x) = x
# _lift_instrinsic(x::Core.IntrinsicFunction) = IntrinsicsWrappers.translate(Val(x))

# _preprocess_expr_arg(ex::Union{Argument, SSAValue, CC.MethodInstance}, _) = ex
# _preprocess_expr_arg(ex::QuoteNode, _) = Literal(_lift_intrinsic(ex.value))

_get_globalref(x::GlobalRef) = getglobal(x.mod, x.name)

# function _preprocess_expr_arg(ex::GlobalRef, _)
#     if _get_globalref(ex) isa Core.IntrinsicFunction
#         return Literal(_lift_intrinsic(_get_globalref(ex)))
#     else
#         return TypedGlobalRef(ex)
#     end
# end

# function _preprocess_expr_arg(ex, _)
#     is_literal(ex) && return Literal(ex)
#     throw(error("$ex of type $(Core.Typeof(ex)) is not a known component of Julia SSAIR"))
# end

# function preprocess_ir(ex::Expr, in_f)
#     ex = CC.copy(ex)
#     sptypes = in_f.sptypes
#     spnames = in_f.spnames
#     if Meta.isexpr(ex, :boundscheck)
#         return Literal(true)
#     elseif Meta.isexpr(ex, :foreigncall)
#         args = ex.args
#         name = extract_foreigncall_name(args[1])
#         sparams_dict = Dict(zip(spnames, sptypes))
#         RT = Val(interpolate_sparams(args[2], sparams_dict))
#         AT = (map(x -> Val(interpolate_sparams(x, sparams_dict)), args[3])..., )
#         nreq = Val(args[4])
#         calling_convention = Val(_extract(args[5]))
#         x = args[6:end]
#         ex.head = :call
#         f = GlobalRef(Taped, :_foreigncall_)
#         ex.args = Any[f, name, RT, AT, nreq, calling_convention, x...]
#         ex.args = map(Base.Fix2(_preprocess_expr_arg, sptypes), ex.args)
#         return ex
#     elseif Meta.isexpr(ex, :new)
#         ex.head = :call
#         ex.args = map(Base.Fix2(_preprocess_expr_arg, sptypes), [_new_, ex.args...])
#         return ex
#     else
#         ex.args = map(Base.Fix2(_preprocess_expr_arg, sptypes), ex.args)
#         return ex
#     end
# end

# @inline _eval(f::F, args::Vararg{Any, N}) where {F, N} = f(args...)

# function build_inst(ir_inst::Expr, in_f, n::Int, b::Int, is_blk_end::Bool)
#     next_blk = _standard_next_block(is_blk_end, b)
#     if Meta.isexpr(ir_inst, :invoke) || Meta.isexpr(ir_inst, :call)

#         # Extract args refs.
#         __args = Meta.isexpr(ir_inst, :invoke) ? ir_inst.args[2:end] : ir_inst.args
#         arg_refs = map(arg -> _get_input(arg, in_f), (__args..., ))

#         ctx = in_f.ctx
#         sig = Tuple{map(eltype, arg_refs)...}
#         evaluator = if is_primitive(ctx, sig)
#             _eval
#         else
#             if all(Base.isconcretetype, sig.parameters)
#                 InterpretedFunction(ctx, sig, in_f.interp)
#             else
#                 DelayedInterpretedFunction{sig, Core.Typeof(ctx)}(ctx, in_f.interp)
#             end
#         end
#         return CallInst(arg_refs, evaluator, in_f.slots[n], next_blk, ir_inst, n)
#     elseif Meta.isexpr(ir_inst, :throw_undef_if_not)
#         val = _get_input(ir_inst.args[2], in_f)
#         return ThrowUndefIfNot(ir_inst.args[1][], val, next_blk)
#     elseif ir_inst.head in [
#         :code_coverage_effect, :gc_preserve_begin, :gc_preserve_end, :loopinfo, :leave,
#         :pop_exception,
#     ]
#         return SkippedExpressionInst(ir_inst.head)
#     else
#         throw(error("Unrecognised expression $ir_inst"))
#     end
# end

# const OC = Core.OpaqueClosure

# function build_coinstructions(ir_inst::Expr, in_f, in_f_rrule!!, n, is_blk_end)
#     is_invoke = Meta.isexpr(ir_inst, :invoke)
#     if is_invoke || Meta.isexpr(ir_inst, :call)

#         # Extract args refs.
#         __args = is_invoke ? ir_inst.args[3:end] : ir_inst.args[2:end]
#         codual_arg_refs = map(arg -> _get_input(arg, in_f_rrule!!), (__args..., ))

#         # Extract val ref.
#         codual_val_ref = in_f_rrule!!.slots[n]

#         # Extract function.
#         fn = is_invoke ? ir_inst.args[2] : ir_inst.args[1]
#         if fn isa Core.SSAValue || fn isa Core.Argument
#             fn = primal(_get_input(fn, in_f_rrule!!)[])
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
#             if !(run_fwds_pass isa FwdsIFInstruction)
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
#             if !(run_rvs_pass isa BwdsIFInstruction)
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

_get_type(x::Core.PartialStruct) = x.typ
_get_type(x::Core.Const) = Core.Typeof(x.val)
_get_type(T) = T

_get_input(x::Core.Argument, _, arg_info) = arg_info.arg_slots[x.n]
_get_input(x::Core.SSAValue, slots, _) = slots[x.id]

# Copied from Umlaut.jl.
extract_foreigncall_name(x::Symbol) = Val(x)
function extract_foreigncall_name(x::Expr)
    # Make sure that we're getting the expression that we're expecting.
    !Meta.isexpr(x, :call) && error("unexpected expr $x")
    !isa(x.args[1], GlobalRef) && error("unexpected expr $x")
    x.args[1].name != :tuple && error("unexpected expr $x")
    length(x.args) != 3 && error("unexpected expr $x")

    # Parse it into a name that can be passed as a type.
    v = eval(x)
    return Val((Symbol(v[1]), Symbol(v[2])))
end
function extract_foreigncall_name(v::Tuple)
    return Val((Symbol(v[1]), Symbol(v[2])))
end
extract_foreigncall_name(x::QuoteNode) = extract_foreigncall_name(x.value)

function sparam_names(m::Core.Method)
    whereparams = ExprTools.where_parameters(m.sig)
    whereparams === nothing && return []
    return map(whereparams) do name
        name isa Symbol && return name
        Meta.isexpr(name, :(<:)) && return name.args[1]
        Meta.isexpr(name, :(>:)) && return name.args[1]
        error("unrecognised type param $name")
    end
end

# Copied from Umlaut.jl. Originally, adapted from
# https://github.com/JuliaDebug/JuliaInterpreter.jl/blob/aefaa300746b95b75f99d944a61a07a8cb145ef3/src/optimize.jl#L239
function interpolate_sparams(@nospecialize(t::Type), sparams::Dict)
    t isa Core.TypeofBottom && return t
    while t isa UnionAll
        t = t.body
    end
    t = t::DataType
    if Base.isvarargtype(t)
        return Expr(:(...), t.parameters[1])
    end
    if Base.has_free_typevars(t)
        params = map(t.parameters) do @nospecialize(p)
            if isa(p, TypeVar)
                return sparams[p.name].typ.val
            elseif isa(p, DataType) && Base.has_free_typevars(p)
                return interpolate_sparams(p, sparams)
            elseif p isa CC.VarState
                @show "doing varstate"
                p.typ
            else
                return p
            end
        end
        T = t.name.Typeofwrapper.parameters[1]
        return T{params...}
    end
    return t
end

#
# Loading arguments into slots.
#

struct ArgInfo{Targ_slots<:NTuple{N, Any} where {N}, is_vararg}
    arg_slots::Targ_slots
end

function arginfo_from_argtypes(::Type{T}, is_vararg::Bool) where {T<:Tuple}
    Targ_slots = Tuple{map(t -> SlotRef{t}, T.parameters)...}
    return ArgInfo{Targ_slots, is_vararg}((map(t -> SlotRef{t}(), T.parameters)..., ))
end

@noinline function load_args!(ai::ArgInfo{T, is_vararg}, args::Tuple) where {T, is_vararg}

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
        num_args = length(ai.arg_slots) - 1 # once for vararg
        refined_args = (args[1:num_args]..., (args[num_args+1:end]..., ))
    else
        refined_args = args
    end

    # Load the arguments into `ai.arg_slots`.
    return __load_args!(ai.arg_slots, refined_args)
end

@generated function __load_args!(arg_slots::Tuple, args::Tuple)
    Ts = args.parameters
    ns = filter(n -> !Base.issingletontype(Ts[n]), eachindex(Ts))
    loaders = map(n -> :(arg_slots[$n][] = args[$n]), ns)
    return Expr(:block, loaders..., :(return nothing))
end

#
# Construct and run an InterpretedFunction.
#

struct InterpretedFunction{sig<:Tuple, C, Treturn, Targ_info<:ArgInfo}
    ctx::C
    return_slot::SlotRef{Treturn}
    arg_info::Targ_info
    slots::Vector{Union{SlotRef, ConstSlot}}
    instructions::Vector{IFInstruction}
    bb_starts::Vector{Int}
    bb_ends::Vector{Int}
    ir::IRCode
    interp::TapedInterpreter
    sparam_names::Any
end

function is_vararg_sig_and_sparam_names(sig)
    world = Base.get_world_counter()
    min = RefValue{UInt}(typemin(UInt))
    max = RefValue{UInt}(typemax(UInt))
    ms = Base._methods_by_ftype(sig, nothing, -1, world, true, min, max, Ptr{Int32}(C_NULL))::Vector
    m = only(ms).method
    return m.isva, sparam_names(m)
end

_get_input(x, in_f::InterpretedFunction) = _get_input(x, in_f.slots, in_f.arg_info)

make_slot(::Type{T}) where {T} = SlotRef{T}()
make_slot(x::CC.Const) = ConstSlot{Core.Typeof(x.val)}(x.val)
make_slot(x::CC.PartialStruct) = SlotRef{x.typ}()

function InterpretedFunction(ctx::C, sig::Type{<:Tuple}, interp) where {C}
    @nospecialize ctx sig

    # If we've already constructed this interpreted function, just return it.
    sig in keys(interp.in_f_cache) && return interp.in_f_cache[sig]

    # Grab code associated to this function.
    output = Base.code_ircode_by_type(sig; interp)
    if isempty(output)
        throw(ArgumentError("No methods found for signature $sig"))
    elseif length(output) > 1
        throw(ArgumentError("$(length(output)) methods found for signature $sig"))
    end
    ir, Treturn = only(output)

    # Slot into which the output of this function will be placed.
    return_slot = SlotRef{Treturn}()

    # Construct argument reference references.
    arg_types = Tuple{map(_get_type, ir.argtypes)..., }
    is_vararg, spnames = is_vararg_sig_and_sparam_names(sig)
    arg_info = arginfo_from_argtypes(arg_types, is_vararg)

    # Create slots. In most cases, these are instances of `SlotRef`s, which can be read from
    # and written to by instructions (they are essentially `Base.RefValue`s with a
    # different name. Very occassionally the compiler will deduce that a particular slot has
    # a constant value. In these cases, we instead create an instance of `ConstSlot`, which
    # cannot be written to.
    slots = AbstractSlot[make_slot(T) for T in ir.stmts.type]

    # Allocate memory for instructions and argument loading instructions.
    insts = Vector{IFInstruction}(undef, length(slots))

    # Compute the index of the instruction associated with the start of each basic block
    # in `ir`. This is used to know where to jump to when we hit a `Core.GotoNode` or
    # `Core.GotoIfNot`. The `ir.cfg` very nearly gives this to us for free.
    bb_starts = vcat(1, ir.cfg.index)
    bb_ends = vcat(ir.cfg.index .- 1, length(slots))

    # Extract the starting location of each basic block from the CFG and build IF.
    in_f = InterpretedFunction{sig, C, Treturn, Core.Typeof(arg_info)}(
        ctx, return_slot, arg_info, slots, insts, bb_starts, bb_ends, ir, interp, spnames,
    )

    # Cache this InterpretedFunction so that we don't have tobuild it again.
    interp.interpreted_function_cache[sig] = in_f

    return in_f
end

load_args!(in_f::InterpretedFunction, args) = load_args!(in_f.arg_info, args)

function (in_f::InterpretedFunction)(args::Vararg{Any, N}) where {N}
    load_args!(in_f, args)
    return __barrier(in_f)
end

function __barrier(in_f::Tf) where {Tf<:InterpretedFunction}
    prev_block = 0
    next_block = 0
    current_block = 1
    n = 1
    instructions = in_f.instructions
    while next_block != -1
        # @show prev_block, current_block, next_block, n
        if !isassigned(instructions, n)
            instructions[n] = build_inst(in_f, n)
        end
        next_block = instructions[n](prev_block)
        if next_block == 0
            n += 1
        elseif next_block > 0
            n = in_f.bb_starts[next_block]
            prev_block = current_block
            current_block = next_block
            next_block = 0
        end
    end
    return in_f.return_slot[]
end

preprocess_ir(st::CC.SSAValue, _) = st
preprocess_ir(st::CC.Argument, _) = st
preprocess_ir(st::CC.QuoteNode, _) = Literal(st.value)
preprocess_ir(st, _) = st

function block_map(cfg::CC.CFG)
    line_to_blk_maps = map(((n, blk),) -> tuple.(blk.stmts, n), enumerate(cfg.blocks))
    return Dict(reduce(vcat, line_to_blk_maps))
end

function build_inst(in_f::InterpretedFunction{sig}, n::Int) where {sig}
    @nospecialize in_f
    d = (sptypes=in_f.ir.sptypes, spnames=in_f.sparam_names)
    ir_inst = preprocess_ir(in_f.ir.stmts.inst[n], d)
    b = block_map(in_f.ir.cfg)[n]
    is_blk_end = n in in_f.bb_ends
    return _make_opaque_closure(build_inst(ir_inst, in_f, n, b, is_blk_end), sig, n)
end

function build_inst(ctx, ir_inst::Any, arg_slots, slots, return_slot, is_block_end)
    println("IR in which error is found:")
    display(sig)
    display(ir)
    println()
    throw(error("unhandled instruction $ir_inst, with type $(typeof(ir_inst))")) 
end

struct DelayedInterpretedFunction{sig, C}
    ctx::C
    interp::TInterp
end

function (din_f::DelayedInterpretedFunction)(fargs...)
    sig = Tuple{map(Core.Typeof, fargs)...}
    if is_primitive(din_f.ctx, sig)
        f, args... = fargs
        return f(args...)
    else
        return InterpretedFunction(din_f.ctx, sig, din_f.interp)(fargs...)
    end
end

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
    fwds_instructions::Vector{FwdsIFInstruction}
    bwds_instructions::Vector{BwdsIFInstruction}
    n_stack::Vector{Int}
end

_get_input(x, in_f::InterpretedFunctionRRule) = _get_input(x, in_f.slots, in_f.arg_info)

function build_rrule!!(in_f::InterpretedFunction{sig}) where {sig}
    return_slot = make_codual_slot(in_f.return_slot)
    arg_info = make_codual_arginfo(in_f.arg_info)
    n_stack = Vector{Int}(undef, 1)
    sizehint!(n_stack, 100)
    return InterpretedFunctionRRule{sig, eltype(return_slot), Core.Typeof(arg_info)}(
        return_slot,
        arg_info,
        map(make_codual_slot, in_f.slots), # SlotRefs
        Vector{FwdsIFInstruction}(undef, length(in_f.instructions)), # fwds_instructions
        Vector{BwdsIFInstruction}(undef, length(in_f.instructions)), # bwds_instructions
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

const __Tinst = Tuple{FwdsIFInstruction, BwdsIFInstruction}

function generate_instructions(in_f, in_f_rrule!!, n)::__Tinst
    d = (sptypes=in_f.ir.sptypes, spnames=in_f.sparam_names)
    ir_inst = preprocess_ir(in_f.ir.stmts.inst[n], in_f)
    is_blk_end = n in in_f.bb_ends
    return build_coinstructions(ir_inst, in_f, in_f_rrule!!, n, is_blk_end)
end

# Slow implementation, but useful for testing correctness.
function rrule!!(f_in::CoDual{<:InterpretedFunction}, args::CoDual...)
    return build_rrule!!(primal(f_in))(f_in, args...)
end

#
# Test cases
#

@noinline function foo(x)
    y = sin(x)
    z = cos(y)
    return z
end

function bar(x, y)
    x1 = sin(x)
    x2 = cos(y)
    x3 = foo(x2)
    x4 = foo(x3)
    x5 = x2 + x4
    return x5
end

const_tester() = cos(5.0)

intrinsic_tester(x) = 5x

function goto_tester(x)
    if x > cos(x)
        @goto aha
    end
    x = sin(x)
    @label aha
    return cos(x)
end

struct Foo
    x::Float64
    y::Symbol
end

new_tester(x, y) = Foo(x, y)

new_tester_2(x) = Foo(x, :symbol)

@eval function new_tester_3(x::Ref{Any})
    y = x[]
    $(Expr(:new, :y, 5.0))
end

__x_for_gref_test = 5.0
@eval globalref_tester() = $(GlobalRef(Taped, :__x_for_gref_test))

function globalref_tester_2(use_gref::Bool)
    v = use_gref ? __x_for_gref_test : 1
    return sin(v)
end

type_unstable_tester(x::Ref{Any}) = cos(x[])

type_unstable_tester_2(x::Ref{Real}) = cos(x[])

type_unstable_function_eval(f::Ref{Any}, x::Float64) = f[](x)

type_unstable_argument_eval(@nospecialize(f), x::Float64) = f(x)

function phi_const_bool_tester(x)
    if x > 0
        a = true
    else
        a = false
    end
    return cos(a)
end

function phi_node_with_undefined_value(x::Bool, y::Float64)
    if x
        v = sin(y)
    end
    z = cos(y)
    if x
        z += v
    end
    return z
end

function pi_node_tester(y::Ref{Any})
    x = y[]
    return isa(x, Int) ? sin(x) : x
end

function avoid_throwing_path_tester(x)
    if x < 0
        Base.throw_boundserror(1:5, 6)
    end
    return sin(x)
end

simple_foreigncall_tester(x) = ccall(:jl_array_isassigned, Cint, (Any, UInt), x, 1)

function simple_foreigncall_tester_2(a::Array{T, M}, dims::NTuple{N, Int}) where {T,N,M}
    ccall(:jl_reshape_array, Array{T,N}, (Any, Any, Any), Array{T,N}, a, dims)
end

function foreigncall_tester(x)
    return ccall(:jl_array_isassigned, Cint, (Any, UInt), x, 1) == 1 ? cos(x[1]) : sin(x[1])
end

function no_primitive_inlining_tester(x)
    X = Matrix{Float64}(undef, 5, 5) # contains a foreigncall which should never be hit
    for n in eachindex(X)
        X[n] = x
    end
    return X
end

@noinline varargs_tester(x::Vararg{Any, N}) where {N} = x

varargs_tester_2(x) = varargs_tester(x)
varargs_tester_2(x, y) = varargs_tester(x, y)
varargs_tester_2(x, y, z) = varargs_tester(x, y, z)

@noinline varargs_tester_3(x, y::Vararg{Any, N}) where {N} = sin(x), y

varargs_tester_4(x) = varargs_tester_3(x...)
varargs_tester_4(x, y) = varargs_tester_3(x...)
varargs_tester_4(x, y, z) = varargs_tester_3(x...)

splatting_tester(x) = varargs_tester(x...)
unstable_splatting_tester(x::Ref{Any}) = varargs_tester(x[]...)

function inferred_const_tester(x::Base.RefValue{Any})
    y = x[]
    y === nothing && return y
    return 5y
end
inferred_const_tester(x::Int) = x == 5 ? x : 5x

a_primitive(x) = sin(x)
non_primitive(x) = sin(x)

is_primitive(::DefaultCtx, ::Type{<:Tuple{typeof(a_primitive), Any}}) = true
is_primitive(::DefaultCtx, ::Type{<:Tuple{typeof(non_primitive), Any}}) = false

contains_primitive(x) = @inline a_primitive(x)
contains_non_primitive(x) = @inline non_primitive(x)
contains_primitive_behind_call(x) = @inline contains_primitive(x)

function to_benchmark(__rrule!!, df, dx)
    out, pb!! = __rrule!!(df, dx...)
    pb!!(tangent(out), tangent(df), map(tangent, dx)...)
end
