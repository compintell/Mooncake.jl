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
    ir_interp::Bool
    function TapedInterpreter(
        ctx::C=DefaultCtx();
        meta=nothing,
        world::UInt=Base.get_world_counter(),
        inf_params::CC.InferenceParams=CC.InferenceParams(),
        opt_params::CC.OptimizationParams=CC.OptimizationParams(),
        inf_cache::Vector{CC.InferenceResult}=CC.InferenceResult[], 
        code_cache::TICache=TICache(),
        ir_interp=false,
    ) where {C}
        return new{C}(
            ctx, meta, world, inf_params, opt_params, inf_cache, code_cache, ir_interp
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
# Special Ref type to avoid confusion between existing ref types.
#

mutable struct SlotRef{T}
    x::T
    SlotRef{T}() where {T} = new{T}()
    SlotRef(x::T) where {T} = new{T}(x)
end

@inline Base.getindex(x::SlotRef) = getfield(x, :x)
@inline function Base.setindex!(x::SlotRef, val)
    setfield!(x, :x, val)
    return x.x
end

extract_arg(x::SlotRef{T}) where {T} = x[]
extract_arg(x::T) where {T} = x # assume literal

extract_codual(x::SlotRef{T}) where {T<:CoDual} = x[]
extract_codual(x::T) where {T} = uninit_codual(x)

Base.isassigned(x::SlotRef) = isdefined(x, :x)

Base.eltype(::SlotRef{T}) where {T} = T

const IFInstruction = Core.OpaqueClosure{Tuple{Int, Int}, Int}
const FwdsIFInstruction = IFInstruction
const BwdsIFInstruction = Core.OpaqueClosure{Tuple{Int}, Int}
const ArgLoadInstruction = Core.OpaqueClosure{Tuple{Int}, Int}

#
# ReturnNode
#

struct ReturnInst{Treturn_slot<:SlotRef, Tval}
    return_slot::Treturn_slot
    val::Tval
end

function (inst::ReturnInst)(::Int, ::Int)
    inst.return_slot[] = extract_arg(inst.val)
    return -1
end

function build_instruction(ir_inst::ReturnNode, in_f, n, is_block_end)
    return ReturnInst(in_f.return_slot, _get_input(ir_inst.val, in_f))
end

function build_coinstructions(ir_inst::ReturnNode, in_f, in_f_rrule!!, n, is_blk_end)
    function __barrier(return_slot::A, slot_to_return::B) where {A, B}
        # Construct operation to run the forwards-pass.
        run_fwds_pass = @opaque function (a::Int, b::Int)
            return_slot[] = extract_codual(slot_to_return)
            return -1
        end
        if !(run_fwds_pass isa FwdsIFInstruction)
            display(CC.code_typed_opaque_closure(run_fwds_pass)[1][1])
            println()
        end

        # Construct operation to run the reverse-pass.
        run_rvs_pass = if slot_to_return isa SlotRef
            @opaque  function (j::Int)
                setfield!(slot_to_return, :x, getfield(return_slot, :x))
                return j
            end
        else
            @opaque (j::Int) -> j
        end
        if !(run_rvs_pass isa BwdsIFInstruction)
            display(CC.code_typed_opaque_closure(run_fwds_pass)[1][1])
            println()
        end
        return run_fwds_pass, run_rvs_pass
    end
    return __barrier(in_f_rrule!!.return_slot, _get_input(ir_inst.val, in_f_rrule!!))
end

#
# GotoNode
#

struct GotoInst
    n::Int
end

(inst::GotoInst)(::Int, ::Int) = inst.n

build_instruction(ir_inst::GotoNode, _, _, _) = GotoInst(ir_inst.label)

function build_coinstructions(ir_inst::GotoNode, in_f, in_f_rrule!!, n, is_blk_end)
    dest = ir_inst.label
    run_fwds_pass::FwdsIFInstruction = @opaque (a::Int, b::Int) -> dest
    run_rvs_pass::BwdsIFInstruction = @opaque (j::Int) -> j
    return run_fwds_pass, run_rvs_pass
end

#
# GotoIfNot
#

struct GotoIfNotInst{Tcond}
    cond::Tcond
    dest::Int
    node::GotoIfNot
    line::Int
end

function (inst::GotoIfNotInst)(::Int, current_block::Int)
    return extract_arg(inst.cond[]) ? current_block + 1 : inst.dest
end

function build_instruction(ir_inst::GotoIfNot, in_f, n, _)
    return GotoIfNotInst(_get_input(ir_inst.cond, in_f), ir_inst.dest, ir_inst, n)
end

function build_coinstructions(ir_inst::GotoIfNot, in_f, in_f_rrule!!, n, is_blk_end)
    cond_slot = _get_input(ir_inst.cond, in_f_rrule!!)
    dest = ir_inst.dest
    run_fwds_pass::FwdsIFInstruction = @opaque function (a::Int, current_block::Int)
        return primal(extract_codual(cond_slot)) ? current_block + 1 : dest
    end
    run_rvs_pass::BwdsIFInstruction = @opaque (j::Int) -> j
    return run_fwds_pass, run_rvs_pass
end

#
# PhiNode
#

# We can always safely assume that all `values` elements are SlotRefs.
struct PhiNodeInst{Tedges<:Tuple, Tvalues<:Tuple, Tval_slot<:SlotRef}
    edges::Tedges
    values::Tvalues
    val_slot::Tval_slot
    node::PhiNode
    line::Int
    is_blk_end::Bool
end

function (inst::PhiNodeInst)(prev_block::Int, current_block::Int)
    for n in eachindex(inst.edges)
        if inst.edges[n] == prev_block
            inst.val_slot[] = extract_arg(inst.values[n])
        end
    end
    return inst.is_blk_end ? current_block + 1 : 0
end

struct UndefinedReference end

function build_instruction(ir_inst::PhiNode, in_f, n::Int, is_blk_end)
    edges = map(Int, (ir_inst.edges..., ))
    values_vec = map(eachindex(ir_inst.values)) do j
        if isassigned(ir_inst.values, j)
            return _get_input(ir_inst.values[j], in_f)
        else
            return UndefinedReference()
        end
    end
    values = map(x -> x isa SlotRef ? x : SlotRef(x), (values_vec..., ))
    val_slot = in_f.slots[n]
    return PhiNodeInst(edges, values, val_slot, ir_inst, n, is_blk_end)
end


function build_coinstructions(ir_inst::PhiNode, _, in_f_rrule!!, n, is_blk_end)

    # Extract relevant values.
    edges = map(Int, (ir_inst.edges..., ))
    values_vec = map(eachindex(ir_inst.values)) do j
        if isassigned(ir_inst.values, j)
            return _get_input(ir_inst.values[j], in_f_rrule!!)
        else
            return UndefinedReference()
        end
    end
    values = map(x -> x isa SlotRef ? x : SlotRef(zero_codual(x)), (values_vec..., ))
    val_slot = in_f_rrule!!.slots[n]

    # Create a value slot stack.
    value_slot_stack = Vector{eltype(val_slot)}(undef, 0)
    prev_block_stack = Vector{Int}(undef, 0)

    # Construct operation to run the forwards-pass.
    run_fwds_pass::FwdsIFInstruction = @opaque function (prev_blk::Int, current_blk::Int)
        push!(prev_block_stack, prev_blk)
        for n in eachindex(edges)
            if edges[n] == prev_blk
                if isassigned(val_slot)
                    push!(value_slot_stack, val_slot[])
                end
                val_slot[] = extract_arg(values[n])
            end
        end
        return is_blk_end ? current_blk + 1 : 0
    end

    # Construct operation to run the reverse-pass.
    run_rvs_pass::BwdsIFInstruction = @opaque function (j::Int)
        prev_block = pop!(prev_block_stack)
        for n in eachindex(edges)
            if edges[n] == prev_block
                replace_tangent!(
                    values[n], increment!!(tangent(values[n][]), tangent(val_slot[])),
                )
                if !isempty(value_slot_stack)
                    val_slot[] = pop!(value_slot_stack)
                end
            end
        end
        return j
    end

    return run_fwds_pass, run_rvs_pass
end

#
# PiNode
#

struct PiNodeInst{Tinput_ref<:SlotRef, Tval_ref<:SlotRef}
    input_ref::Tinput_ref
    val_ref::Tval_ref
    is_blk_end::Bool
end

function (inst::PiNodeInst)(::Int, current_block::Int)
    inst.val_ref[] = inst.input_ref[]
    return inst.is_blk_end ? current_block + 1 : 0
end

function build_instruction(ir_inst::PiNode, in_f, n, is_blk_end)
    return PiNodeInst(_get_input(ir_inst.val, in_f), in_f.slots[n], is_blk_end)
end

function build_coinstructions(ir_inst::PiNode, _, in_f_rrule!!, n::Int, is_blk_end::Bool)
    input_ref = _get_input(ir_inst.val, in_f_rrule!!)
    val_ref = in_f_rrule!!.slots[n]
    run_fwds_pass::FwdsIFInstruction = @opaque function(::Int, current_blk::Int)
        val_ref[] = input_ref[]
        return is_blk_end ? current_blk + 1 : 0
    end
    run_rvs_pass::BwdsIFInstruction = @opaque function(j::Int)
        input_ref[] = val_ref[]
        return j
    end
    return run_fwds_pass, run_rvs_pass
end

#
# Nothing
#

struct NothingInst end

(inst::NothingInst)(::Int, current_block::Int) = current_block + 1

build_instruction(::Nothing, in_f, n, is_block_end) = NothingInst()

function build_coinstructions(::Nothing, _, _, _, _)
    run_fwds_pass = @opaque (a::Int, current_block::Int) -> current_block + 1
    run_rvs_pass = @opaque (j::Int) -> j
    return run_fwds_pass, run_rvs_pass
end

#
# Bool
#

struct BoolInst
    val::Bool
    val_ref::SlotRef{Bool}
    is_blk_end::Bool
end

function (inst::BoolInst)(::Int, current_block::Int)
    inst.val_ref[] = inst.val
    return inst.is_blk_end ? current_block + 1 : 0
end

build_instruction(val::Bool, in_f, n, is_blk_end) = BoolInst(val, in_f.slots[n], is_blk_end)

function build_coinstructions(ir_inst::Bool, _, in_f_rrule!!, n, is_blk_end)
    function __barrier(val, val_ref::SlotRef{<:CoDual{Bool}}, is_blk_end)
        run_fwds_pass::FwdsIFInstruction = @opaque function(a::Int, current_block::Int)
            val_ref[] = zero_codual(val)
            return is_blk_end ? current_block + 1 : 0
        end
        run_rvs_pass::BwdsIFInstruction = @opaque (j::Int) -> j
        return run_fwds_pass, run_rvs_pass
    end
    return __barrier(ir_inst, in_f_rrule!!.slots[n], is_blk_end)
end

#
# Type
#

struct TypeInst{T}
    val::Type
    val_ref::SlotRef{T}
    is_blk_end::Bool
end

function (inst::TypeInst)(::Int, current_block::Int)
    inst.val_ref[] = inst.val
    return inst.is_blk_end ? current_block + 1 : 0
end

build_instruction(val::Type, in_f, n, is_blk_end) = TypeInst(val, in_f.slots[n], is_blk_end)

function build_coinstructions(val::Type, _, in_f_rrule!!, n, is_blk_end)
    function __barrier(val, val_ref::SlotRef{<:CoDual{<:Type}}, is_blk_end)
        run_fwds_pass::FwdsIFInstruction = @opaque function(a::Int, current_block::Int)
            val_ref[] = zero_codual(val)
            return is_blk_end ? current_block + 1 : 0
        end
        run_rvs_pass::BwdsIFInstruction = @opaque (j::Int) -> j
        return run_fwds_pass, run_rvs_pass
    end
    return __barrier(val, in_f_rrule!!.slots[n], is_blk_end)
end

#
# GlobalRef
#

struct GlobalRefInst{Tc, Tval_ref}
    c::Tc
    val_ref::Tval_ref
    is_blk_end::Bool
end

function (inst::GlobalRefInst)(::Int, current_block::Int)
    inst.val_ref[] = inst.c
    return inst.is_blk_end ? current_block + 1 : 0
end

function build_instruction(node::GlobalRef, in_f, n, is_blk_end)
    return GlobalRefInst(_get_globalref(node), in_f.slots[n], is_blk_end)
end

function build_coinstructions(node::GlobalRef, _, in_f_rrule!!, n, is_blk_end)
    function __barrier(c::A, val_ref::B) where {A, B}
        run_fwds_pass = @opaque function (a::Int, current_blk::Int)
            val_ref[] = c
            return is_blk_end ? current_blk + 1 : 0
        end
        run_rvs_pass = @opaque (j::Int) -> j
        return run_fwds_pass, run_rvs_pass
    end
    return __barrier(uninit_codual(_get_globalref(node)), in_f_rrule!!.slots[n])
end

#
# Expr -- this is a big one
#

struct CallInst{Tf, Targs, Tval_ref<:SlotRef}
    f::Tf
    args::Targs
    val_ref::Tval_ref
    ir::Expr
    line::Int
    is_blk_end::Bool
end

function (inst::CallInst{sig, A})(::Int, current_block::Int) where {sig, A}
    # display(("call", inst.line, inst.ir, inst.args))
    args = map(extract_arg, inst.args)
    inst.val_ref[] = inst.f(args...)
    return inst.is_blk_end ? current_block + 1 : 0
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

# Handles the case where `x` is a constant, rather than a slot.
replace_tangent!(x, new_tangent) = nothing

struct SkippedExpressionInst
    s::Symbol
end

(::SkippedExpressionInst)(::Int, ::Int) = 0

function build_instruction(ir_inst::Expr, in_f, n::Int, is_block_end::Bool)
    ctx = in_f.ctx
    arg_info = in_f.arg_info
    slots = in_f.slots
    is_invoke = Meta.isexpr(ir_inst, :invoke)
    if is_invoke || Meta.isexpr(ir_inst, :call)

        # Extract args refs.
        __args = is_invoke ? ir_inst.args[3:end] : ir_inst.args[2:end]
        arg_refs = map(arg -> _get_input(arg, slots, arg_info), (__args..., ))

        # Extract val ref.
        val_ref = slots[n]

        # Extract function.
        fn = is_invoke ? ir_inst.args[2] : ir_inst.args[1]

        if fn isa Core.SSAValue || fn isa Core.Argument
            fn = _get_input(fn, slots, arg_info)[]
        end
        if fn isa GlobalRef
            fn = getglobal(fn.mod, fn.name)
        end
        if fn isa Core.IntrinsicFunction
            fn = IntrinsicsWrappers.translate(Val(fn))
        end

        fn_sig = Tuple{Core.Typeof(fn), map(_robust_typeof, arg_refs)...}
        if !is_primitive(ctx, fn_sig)
            if all(Base.isconcretetype, fn_sig.parameters)
                fn = InterpretedFunction(ctx, fn_sig; interp=in_f.interp)
            else
                fn = DelayedInterpretedFunction{Core.Typeof(ctx), Core.Typeof(fn)}(
                    ctx, fn, in_f.interp
                )
            end
        end
        new_inst = CallInst(fn, arg_refs, val_ref, ir_inst, n, is_block_end)
        return new_inst
    elseif ir_inst.head in [
        :code_coverage_effect, :gc_preserve_begin, :gc_preserve_end, :loopinfo, :leave,
        :pop_exception,
    ]
        return SkippedExpressionInst(ir_inst.head)
    else
        throw(error("Unrecognised expression $ir_inst"))
    end
end

const OC = Core.OpaqueClosure

function build_coinstructions(ir_inst::Expr, in_f, in_f_rrule!!, n, is_blk_end)
    is_invoke = Meta.isexpr(ir_inst, :invoke)
    if is_invoke || Meta.isexpr(ir_inst, :call)

        # Extract args refs.
        __args = is_invoke ? ir_inst.args[3:end] : ir_inst.args[2:end]
        codual_arg_refs = map(arg -> _get_input(arg, in_f_rrule!!), (__args..., ))

        # Extract val ref.
        codual_val_ref = in_f_rrule!!.slots[n]

        # Extract function.
        fn = is_invoke ? ir_inst.args[2] : ir_inst.args[1]
        if fn isa Core.SSAValue || fn isa Core.Argument
            fn = primal(_get_input(fn, in_f_rrule!!)[])
        end
        if fn isa GlobalRef
            fn = getglobal(fn.mod, fn.name)
        end
        if fn isa Core.IntrinsicFunction
            fn = IntrinsicsWrappers.translate(Val(fn))
        end

        # Check whether any arguments are duplicated. If they are, we must replace the
        # function with a call to dedup.

        fn_sig = Tuple{
            Core.Typeof(fn),
            map(_robust_typeof ∘ primal ∘ extract_codual, codual_arg_refs)...,
        }
        __rrule!! = rrule!!
        # @show fn_sig, is_primitive(in_f.ctx, fn_sig)
        # @show Core.Typeof(fn_sig)
        # @show Base.which(is_primitive, Tuple{MinimalCtx, Core.Typeof(fn_sig)})
        if !is_primitive(in_f.ctx, fn_sig)
            if all(Base.isconcretetype, fn_sig.parameters)
                fn = InterpretedFunction(in_f.ctx, fn_sig; interp=in_f.interp)
                __rrule!! = build_rrule!!(fn)
            else
                fn = DelayedInterpretedFunction{Core.Typeof(ctx), Core.Typeof(fn)}(
                    ctx, fn, in_f.interp
                )
            end
        end

        # Wrap f to make it rrule!!-friendly.
        fn = uninit_codual(fn)

        # Create stacks for storing intermediates.
        codual_sig = Tuple{
            Core.Typeof(fn), map(_robust_typeof ∘ extract_codual, codual_arg_refs)...
        }
        output = Base.return_types(__rrule!!, codual_sig)
        if length(output) == 0
            throw(error("No return type inferred for __rrule!! with sig $codual_sig"))
        elseif length(output) > 1
            @warn "Too many output types inferred"
            display(output)
            println()
            throw(error("> 1 return type inferred for __rrule!! with sig $codual_sig "))
        end
        T_pb!! = only(output)
        if T_pb!! <: Tuple && T_pb!! !== Union{}
            pb_stack = Vector{T_pb!!.parameters[2]}(undef, 0)
        else
            pb_stack = Vector{Any}(undef, 0)
        end
        sizehint!(pb_stack, 100)
        old_vals = Vector{eltype(codual_val_ref)}(undef, 0)
        sizehint!(old_vals, 100)


        function __barrier(fn, codual_val_ref, __rrule!!, old_vals, pb_stack, is_blk_end)

            @noinline function ___fwds_pass(
                codual_val_ref, old_vals, fn, pb_stack, current_blk
            )
                if isassigned(codual_val_ref)
                    push!(old_vals, codual_val_ref[])
                end
                out, pb!! = __rrule!!(fn, map(extract_codual, codual_arg_refs)...)
                codual_val_ref[] = out
                push!(pb_stack, pb!!)
                return is_blk_end ? current_blk + 1 : 0
            end

            # Construct operation to run the forwards-pass.
            run_fwds_pass = @opaque function (a::Int, current_blk::Int)
                ___fwds_pass(codual_val_ref, old_vals, fn, pb_stack, current_blk)
            end
            if !(run_fwds_pass isa FwdsIFInstruction)
                @warn "Unable to compiled forwards pass -- running to generate the error."
                @show run_fwds_pass(5, 4)
            end

            # Construct operation to run the reverse-pass.
            run_rvs_pass = @opaque function (j::Int)
                dout = tangent(codual_val_ref[])
                dargs = map(tangent, map(extract_codual, codual_arg_refs))
                _, new_dargs... = pop!(pb_stack)(dout, tangent(fn), dargs...)
                map(replace_tangent!, codual_arg_refs, new_dargs)
                if !isempty(old_vals)
                    codual_val_ref[] = pop!(old_vals) # restore old state.
                end
                return j
            end
            if !(run_rvs_pass isa BwdsIFInstruction)
                @warn "Unable to compiled reverse pass -- running to generate the error."
                @show run_reverse_pass(5)
            end

            return run_fwds_pass, run_rvs_pass
        end
        return __barrier(fn, codual_val_ref, __rrule!!, old_vals, pb_stack, is_blk_end)
    elseif ir_inst isa Expr && ir_inst.head in [
        :code_coverage_effect, :gc_preserve_begin, :gc_preserve_end, :loopinfo,
        :leave, :pop_exception,
    ]
        run_fwds_pass = @opaque (a::Int, b::Int) -> 0
        run_rvs_pass = @opaque (j::Int) -> j
        return run_fwds_pass, run_rvs_pass
    else
        throw(error("Unrecognised expression $ir_inst"))
    end
end

#
# Code execution
#

_robust_typeof(x::T) where {T} = T
_robust_typeof(::Type{T}) where {T} = Type{T}
_robust_typeof(x::SlotRef{T}) where {T} = T

_other_robust_typeof(x::T) where {T} = T
_other_robust_typeof(::Type{T}) where {T} = Type{T}
_other_robust_typeof(x::SlotRef{T}) where {T} = SlotRef{T}

_get_type(x::Core.PartialStruct) = x.typ
_get_type(x::Core.Const) = _robust_typeof(x.val)
_get_type(T) = T

_get_globalref(x::GlobalRef) = getglobal(x.mod, x.name)

_deref(x::GlobalRef) = _get_globalref(x)
_deref(::Type{T}) where {T} = T

_get_input(x::QuoteNode, _, _) = x.value
_get_input(x::GlobalRef, _, _) = _get_globalref(x)
_get_input(x::Core.SSAValue, slots, _) = slots[x.id]
_get_input(x::Core.Argument, _, arg_info) = arg_info.arg_slots[x.n]
_get_input(x, _, _) = x

function _make_opaque_closure(inst, sig, n)
    oc = @opaque Tuple{Int, Int} (p, q) -> inst(p, q)
    if !(oc isa IFInstruction)
        println("Displaying debugging info from _make_opaque_closure:")
        println("sig of InterpretedFunction in which this is instruction $n:")
        display(sig)
        println()
        println("inst:")
        display(inst)
        println()
        @show oc(5, 4)
    end
    return oc::IFInstruction
end

get_tuple_type(x::Tuple) = Tuple{map(_other_robust_typeof, x)...}

_get_arg_type(::Type{Val{T}}) where {T} = T

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
                return sparams[p.name]
            elseif isa(p, DataType) && Base.has_free_typevars(p)
                return interpolate_sparams(p, sparams)
            else
                return p
            end
        end
        T = t.name.Typeofwrapper.parameters[1]
        return T{params...}
    end
    return t
end


_extract(x::Symbol) = x
_extract(x::QuoteNode) = x.value

function rewrite_special_cases(ir::IRCode, st::Expr)
    st = CC.copy(st)
    ex = Meta.isexpr(st, :(=)) ? st.args[2] : st
    if Meta.isexpr(ex, :boundscheck)
        ex = true
    end
    if ex isa Expr
        ex.args = [Meta.isexpr(arg, :boundscheck) ? true : arg for arg in ex.args]
    end
    if Meta.isexpr(ex, :foreigncall)
        args = st.args
        name = extract_foreigncall_name(args[1])
        RT = Val(args[2])
        AT = (map(Val, args[3])..., )
        # RT = Val(interpolate_sparams(args[2], sparams_dict))
        # AT = (map(x -> Val(interpolate_sparams(x, sparams_dict)), args[3])..., )
        nreq = Val(args[4])
        calling_convention = Val(_extract(args[5]))
        x = args[6:end]
        ex.head = :call
        ex.args = Any[_foreigncall_, name, RT, AT, nreq, calling_convention, x...]
    end
    if Meta.isexpr(ex, :new)
        ex.head = :call
        T = if ex.args[1] isa GlobalRef
            _get_globalref(ex.args[1])
        elseif ex.args[1] isa Type
            ex.args[1]
        else
            throw(error("type is $(ex.args[1]), of type $(Core.Typeof(ex.args[1]))"))
        end
        ex.args = Any[New{T}(), ex.args[2:end]...]
    end
    if Meta.isexpr(ex, :static_parameter)
        ex = ir.sptypes[ex.args[1]]
        if ex isa CC.VarState
            ex = ex.typ
        end
    end
    return Meta.isexpr(st, :(=)) ? Expr(:(=), st.args[1], ex) : ex
end
rewrite_special_cases(::IRCode, st) = st
function rewrite_special_cases(ir::IRCode, st::GotoIfNot)
    return GotoIfNot(rewrite_special_cases(ir, st.cond), st.dest)
end

#
# Loading arguments into slots.
#

struct ArgInfo{Targ_slots<:Tuple, is_vararg}
    arg_slots::Targ_slots
end

function arginfo_from_argtypes(::Type{T}, is_vararg::Bool) where {T<:Tuple}
    Targ_slots = Tuple{map(t -> SlotRef{t}, T.parameters)...}
    return ArgInfo{Targ_slots, is_vararg}((map(t -> SlotRef{t}(), T.parameters)..., ))
end

function load_args!(ai::ArgInfo{T, is_vararg}, args::Tuple) where {T, is_vararg}

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
        refined_args = (args[1:num_args]..., (args[num_args+1:end]..., ))
    else
        refined_args = args
    end

    # Load the arguments into `ai.arg_slots`.
    return __load_args!(ai.arg_slots, refined_args)
end

@generated function __load_args!(arg_slots::Tuple, args::Tuple)
    return Expr(
        :block,
        map(n -> :(arg_slots[$(n + 1)][] = args[$n]), eachindex(args.parameters))...,
        :(return nothing),
    )
end

#
# Construct and run an InterpretedFunction.
#

struct InterpretedFunction{sig<:Tuple, C, Treturn, Targ_info<:ArgInfo}
    ctx::C
    return_slot::SlotRef{Treturn}
    arg_info::Targ_info
    slots::Vector{SlotRef}
    instructions::Vector{IFInstruction}
    bb_starts::Vector{Int}
    bb_ends::Vector{Int}
    ir::IRCode
    interp::TapedInterpreter
end

function is_vararg_sig(sig)
    world = Base.get_world_counter()
    min = RefValue{UInt}(typemin(UInt))
    max = RefValue{UInt}(typemax(UInt))
    ms = Base._methods_by_ftype(sig, nothing, -1, world, true, min, max, Ptr{Int32}(C_NULL))::Vector
    m = only(ms).method
    return m.isva
end

_get_input(x, in_f::InterpretedFunction) = _get_input(x, in_f.slots, in_f.arg_info)

const __InF_TABLE = Dict{Any, InterpretedFunction}()

"""
    flush_interpreted_function_cache!()

If you modify any code that you have previously run as an `InterpretedFunction`, you should
run this function to clear the cache -- in general, use this function liberally.
"""
function flush_interpreted_function_cache!()
    empty!(__InF_TABLE)
    return nothing
end

function InterpretedFunction(ctx::C, sig::Type{<:Tuple}; interp) where {C}
    @nospecialize ctx sig

    if sig in keys(__InF_TABLE)
        return __InF_TABLE[sig]
    end

    # Grab code associated to this function.
    output = Base.code_ircode_by_type(sig; interp)
    if isempty(output)
        throw(ArgumentError("No methods found for signature $sig"))
    elseif length(output) > 1
        @warn "found multiple methods:"
        display(output)
        println()
        throw(ArgumentError("Multiple methods found for signature $sig"))
    end
    ir, Treturn = only(output)

    # Slot into which the output of this function will be placed.
    return_slot = SlotRef{Treturn}()

    # Construct argument reference references.
    arg_types = Tuple{map(_get_type, ir.argtypes)..., }
    is_vararg = is_vararg_sig(sig)
    arg_info = arginfo_from_argtypes(arg_types, is_vararg)

    # Extract slots.
    slots = SlotRef[SlotRef{_get_type(T)}() for T in ir.stmts.type]

    # Allocate memory for instructions and argument loading instructions.
    instructions = Vector{IFInstruction}(undef, length(slots))

    # Compute the index of the instruction associated with the start of each basic block
    # in `ir`. This is used to know where to jump to when we hit a `Core.GotoNode` or
    # `Core.GotoIfNot`. The `ir.cfg` very nearly gives this to us for free.
    bb_starts = vcat(1, ir.cfg.index)
    bb_ends = vcat(ir.cfg.index .- 1, length(slots))

    # Extract the starting location of each basic block from the CFG and build IF.
    in_f = InterpretedFunction{sig, C, Treturn, Core.Typeof(arg_info)}(
        ctx, return_slot, arg_info, slots, instructions, bb_starts, bb_ends, ir, interp
    )

    __InF_TABLE[sig] = in_f

    return in_f
end

function (in_f::InterpretedFunction{sig})(args::Vararg{Any, N}) where {N, sig}
    @nospecialize in_f, args
    load_args!(in_f.arg_info, args)
    prev_block = 0
    next_block = 0
    current_block = 1
    n = 1
    instructions = in_f.instructions
    while next_block != -1
        # @show prev_block, current_block, next_block, n
        if !isassigned(instructions, n)
            instructions[n] = build_instruction(in_f, n)
        end
        next_block = instructions[n](prev_block, current_block)
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

function build_instruction(in_f::InterpretedFunction{sig}, n::Int) where {sig}
    @nospecialize in_f
    ir_inst = rewrite_special_cases(in_f.ir, in_f.ir.stmts.inst[n])
    is_blk_end = n in in_f.bb_ends
    return _make_opaque_closure(build_instruction(ir_inst, in_f, n, is_blk_end), sig, n)
end

function build_instruction(ctx, ir_inst::Any, arg_slots, slots, return_slot, is_block_end)
    println("IR in which error is found:")
    display(sig)
    display(ir)
    println()
    throw(error("unhandled instruction $ir_inst, with type $(typeof(ir_inst))")) 
end

struct DelayedInterpretedFunction{C, F}
    ctx::C
    f::F
    interp::TInterp
end

function (f::DelayedInterpretedFunction{C, F})(args...) where {C, F}
    s = Tuple{F, map(Core.Typeof, args)...}
    if is_primitive(f.ctx, s)
        return f.f(args...)
    else
        return InterpretedFunction(f.ctx, s; interp=f.interp)(args...)
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
            in_f.instructions[n] = build_instruction(in_f, n)
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
    ir_inst = rewrite_special_cases(in_f.ir, in_f.ir.stmts.inst[n])
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

new_2_tester(x) = Foo(x, :symbol)

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

function foreigncall_tester(x)
    if ccall(:jl_array_isassigned, Cint, (Any, UInt), x, 1) == 1
        return cos(x[1])
    else
        return sin(x[1])
    end
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
