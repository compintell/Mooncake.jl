mutable struct SlotRef{T}
    x::T
    SlotRef{T}() where {T} = new{T}()
    SlotRef(x::T) where {T} = new{T}(x)
end

@inline Base.getindex(x::SlotRef) = x.x
@inline function Base.setindex!(x::SlotRef, val)
    setfield!(x, :x, val)
    return x.x
end

@inline extract_arg(x::SlotRef{T}) where {T} = x[]
@inline extract_arg(x::T) where {T} = x # assume literal

struct InterpretedFunction{sig, Treturn_slot<:SlotRef, Targ_slots, Tslots}
    return_slot::Treturn_slot
    arg_slots::Targ_slots
    slots::Tslots
    instructions::Vector{Core.OpaqueClosure{Tuple{Int, Int}, Int}}
    bb_starts::Vector{Int}
end

mutable struct LazyInterpretedFunction{sig, Treturn_slot, Targ_slots, Tslots}
    f::InterpretedFunction{sig, Treturn_slot, Targ_slots, Tslots}
    LazyInterpretedFunction{sig, A, B, C}() where {sig, A, B, C} = new{sig, A, B, C}()
end

function LazyInterpretedFunction(sig::Type{<:Tuple})
    ir, Treturn = Base.code_ircode_by_type(sig)[1]

    # Construct return slot.
    return_slot = SlotRef{Treturn}()
    Treturn_slot = SlotRef{_robust_typeof(return_slot)}

    # Construct argument reference references.
    arg_slots = map(T -> SlotRef{_get_type(T)}(), (ir.argtypes..., ))
    Targ_slots = Core.Typeof(arg_slots)

    # Extract slot types.
    slots = map(T -> SlotRef{_get_type(T)}(), tuple(ir.stmts.type...))
    Tslots = Core.Typeof(slots)

    return LazyInterpretedFunction{sig, Treturn_slot, Targ_slots, Tslots}()
end

function (f::LazyInterpretedFunction{sig})(args...) where {sig}
    !isdefined(f, :f) && setfield!(f, :f, InterpretedFunction(sig))
    return f.f(args...)
end

struct DelayedInterpretedFunction{F}
    f::F
end

function (f::DelayedInterpretedFunction{F})(args...) where {F}
    sig = Tuple{F, typeof(args).parameters...}
    if is_primitive(sig)
        return f.f(args...)
    else
        return InterpretedFunction(sig)(args...)
    end
end


#
# Basic function evaluation.
#

struct CallInst{Tf, Targs, Tval_ref<:SlotRef}
    f::Tf
    args::Targs
    val_ref::Tval_ref
end

function (inst::CallInst)(::Int, ::Int)
    inst.val_ref[] = inst.f(map(extract_arg, inst.args)...)
    return 0
end

struct NewInst{T, Targs, Tval_ref<:SlotRef}
    args::Targs
    val_ref::Tval_ref
    function NewInst{Tf}(args::Targs, val_ref::Tval_ref) where {Tf, Targs, Tval_ref}
        return new{Tf, Targs, Tval_ref}(args, val_ref)
    end
end

@generated function (inst::NewInst{T, Targs})(::Int, ::Int) where {T, Targs}
    nargs = length(Targs.parameters)
    return Expr(
        :block,
        Expr(
            :(=),
            :(inst.val_ref[]),
            Expr(:new, T, map(n -> :(inst.args[$n][]), 1:nargs)...),
        ),
        :(return 0),
    )
    inst.val_ref[] = :(Expr(:new, T, ))
end

#
# Control-flow instructions. :enter not currently implemented.
#

# Returning -1 indicates that the calling function should return.
struct ReturnInst{Treturn_slot<:SlotRef, Tval}
    return_slot::Treturn_slot
    val::Tval
end

function (inst::ReturnInst)(::Int, ::Int)
    inst.return_slot[] = extract_arg(inst.val)
    return -1
end

struct GotoInst
    n::Int
end

(inst::GotoInst)(::Int, ::Int) = inst.n

struct GotoIfNotInst{Tcond}
    cond::Tcond
    dest::Int
end

function (inst::GotoIfNotInst)(::Int, current_block::Int)
    return extract_arg(inst.cond[]) ? current_block + 1 : inst.dest
end

#
# Other items which can appear inside basic blocks.
# In order to implement this properly, I need to know which block I came from, which isn't
# something that I'm currently doing. In turn, this means that _everything_ must be done in
# terms of basic blocks, rather than line indices. This is fine, but requires some care.
#

struct PhiNodeInst{Tedges<:Tuple, Tvalues<:Tuple, Tval_slot<:SlotRef}
    edges::Tedges
    values::Tvalues
    val_slot::Tval_slot
end

function (inst::PhiNodeInst)(prev_block::Int, ::Int)
    vals = map(extract_arg, inst.values) # extract all for type stability
    for n in eachindex(inst.edges)
        if inst.edges[n] == prev_block
            inst.val_slot[] = vals[n]
        end
    end
    return 0
end

struct PiNodeInst{Tinput_ref<:SlotRef, Tval_ref<:SlotRef}
    input_ref::Tinput_ref
    val_ref::Tval_ref
end

function (inst::PiNodeInst)(::Int, ::Int)
    inst.val_ref[] = inst.input_ref[]
    return 0
end

struct NothingInst end

(inst::NothingInst)(::Int, current_block::Int) = current_block + 1

struct ConstLikeInst{Tc, Tval_ref}
    c::Tc
    val_ref::Tval_ref
end

function (inst::ConstLikeInst)(::Int, ::Int)
    inst.val_ref[] = inst.c
    return 0
end

struct UndefinedReference end



is_primitive(::Type) = false
is_primitive(::Type{Tuple{typeof(sin), Float64}}) = true
is_primitive(::Type{Tuple{typeof(cos), Float64}}) = true
is_primitive(::Type{<:Tuple{Core.Builtin, Vararg{Any, N}}}) where {N} = true

_robust_typeof(x::T) where {T} = T
_robust_typeof(::Type{T}) where {T} = Type{T}
_robust_typeof(x::SlotRef{T}) where {T} = T

_other_robust_typeof(x::T) where {T} = T
_other_robust_typeof(::Type{T}) where {T} = Type{T}
_other_robust_typeof(x::SlotRef{T}) where {T} = SlotRef{T}

_get_type(x::Core.PartialStruct) = x.typ
_get_type(x::Core.Const) = _robust_typeof(x.val)
_get_type(T) = T

get_ir(inst) = Base.code_ircode_by_type(Tuple{Core.Typeof(inst), Int, Int})[1][1]

_get_globalref(x::GlobalRef) = getglobal(x.mod, x.name)

_deref(x::GlobalRef) = _get_globalref(x)
_deref(::Type{T}) where {T} = T

_get_input(x::QuoteNode, slots, arg_slots) = x.value
_get_input(x::GlobalRef, slots, arg_slots) = _get_globalref(x)
_get_input(x::Core.SSAValue, slots, arg_slots) = slots[x.id]
_get_input(x::Core.Argument, slots, arg_slots) = arg_slots[x.n]
_get_input(x, _, _) = x

function _make_opaque_closure(inst, sig)
    new_ir = get_ir(inst)
    empty!(new_ir.argtypes)
    push!(new_ir.argtypes, Core.Typeof(inst))
    push!(new_ir.argtypes, Int)
    push!(new_ir.argtypes, Int)
    oc = Core.OpaqueClosure(new_ir, inst)
    if !(oc isa Core.OpaqueClosure{Tuple{Int, Int}, Int})
        display(sig)
        display(inst)
        display(new_ir)
        println()
    end
    return oc::Core.OpaqueClosure{Tuple{Int, Int}, Int}
end

get_tuple_type(x::Tuple) = Tuple{map(_other_robust_typeof, x)...}

function rewrite_special_cases(st::Expr)
    ex = Meta.isexpr(st, :(=)) ? st.args[2] : st
    if Meta.isexpr(ex, :boundscheck)
        ex = true
    end
    if ex isa Expr
        ex.args = [Meta.isexpr(arg, :boundscheck) ? true : arg for arg in ex.args]
    end
    return Meta.isexpr(st, :(=)) ? Expr(:(=), st.args[1], ex) : ex
end
rewrite_special_cases(st) = st
function rewrite_special_cases(st::Core.GotoIfNot)
    return Core.GotoIfNot(rewrite_special_cases(st.cond), st.dest)
end

function InterpretedFunction(sig::Type{<:Tuple})

    # Grab code associated to this function.
    ir, Treturn = Base.code_ircode_by_type(sig)[1]

    # Construct return slot.
    return_slot = SlotRef{Treturn}()
    Treturn_slot = SlotRef{_robust_typeof(return_slot)}

    # Construct argument reference references.
    arg_slots = map(T -> SlotRef{_get_type(T)}(), (ir.argtypes..., ))
    Targ_slots = Core.Typeof(arg_slots)

    # Extract slot types.
    slots = map(T -> SlotRef{_get_type(T)}(), tuple(ir.stmts.type...))
    Tslots = Core.Typeof(slots)

    # Construct instructions (we should ideally do this recursively, but can't yet).
    instructions = map(eachindex(slots)) do n
        ir_inst = ir.stmts.inst[n]
        ir_inst = rewrite_special_cases(ir_inst)

        is_invoke = Meta.isexpr(ir_inst, :invoke)
        inst = if is_invoke || Meta.isexpr(ir_inst, :call)

            # Extract args / arg pointers.
            __args = is_invoke ? ir_inst.args[3:end] : ir_inst.args[2:end]
            arg_refs = map(arg -> _get_input(arg, slots, arg_slots), (__args..., ))

            # Extract val pointer.
            val_ref = slots[n]

            # Extract function.
            fn = is_invoke ? ir_inst.args[2] : ir_inst.args[1]
            if fn isa GlobalRef
                fn = getglobal(fn.mod, fn.name)
            end
            if fn isa Core.IntrinsicFunction
                fn = IntrinsicsWrappers.translate(Val(fn))
            end

            fn_sig = Tuple{typeof(fn), map(_robust_typeof, arg_refs)...}
            if !is_primitive(fn_sig)
                if isconcretetype(fn_sig)
                    fn = LazyInterpretedFunction(fn_sig)
                else
                    fn = DelayedInterpretedFunction(fn)
                end
            end

            CallInst(fn, arg_refs, val_ref)
        elseif ir_inst isa Core.ReturnNode
            arg_slot = _get_input(ir_inst.val, slots, arg_slots)
            ReturnInst(return_slot, arg_slot)
        elseif ir_inst isa Core.GotoIfNot
            GotoIfNotInst(_get_input(ir_inst.cond, slots, arg_slots), ir_inst.dest)
        elseif ir_inst isa Core.GotoNode
            GotoInst(ir_inst.label)
        elseif ir_inst isa Core.PhiNode
            edges = map(Int, (ir_inst.edges..., ))
            values_vec = map(eachindex(ir_inst.values)) do j
                if isassigned(ir_inst.values, j)
                    return _get_input(ir_inst.values[j], slots, arg_slots)
                else
                    return UndefinedReference()
                end
            end
            values = (values_vec..., )
            val_slot = slots[n]
            PhiNodeInst(edges, values, val_slot)
        elseif ir_inst isa Nothing
            NothingInst()
        elseif Meta.isexpr(ir_inst, :new)
            __args = (ir_inst.args[2:end]..., )
            arg_refs = map(arg -> _get_input(arg, slots, arg_slots), __args)
            val_ref = slots[n]
            NewInst{_deref(ir_inst.args[1])}(arg_refs, val_ref)
        elseif ir_inst isa Core.PiNode
            input_ref = _get_input(ir_inst.val, slots, arg_slots)
            PiNodeInst(input_ref, slots[n])
        elseif ir_inst isa GlobalRef
            ConstLikeInst(_get_globalref(ir_inst), slots[n])
        else
            println("IR in which error is found:")
            display(sig)
            display(ir)
            println()
            throw(error("unhandled instruction $ir_inst, with type $(typeof(ir_inst))")) 
        end
        return _make_opaque_closure(inst, sig)
    end

    # Extract the starting location of each basic block from the CFG and build IF.
    return InterpretedFunction{sig, Treturn_slot, Targ_slots, Tslots}(
        return_slot, arg_slots, slots, instructions, vcat(1, ir.cfg.index)
    )
end

function (in_f::InterpretedFunction)(args::Vararg{Any, N}) where {N}
    load_args!(in_f, args)
    prev_block = 0
    next_block = 0
    current_block = 1
    n = 1
    while next_block != -1
        # @show prev_block, next_block, current_block, n
        next_block = in_f.instructions[n](prev_block, current_block)
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

@generated function load_args!(in_f, args::Tuple)
    return Expr(
        :block,
        map(n -> :(in_f.arg_slots[$(n + 1)][] = args[$n]), eachindex(args.parameters))...,
        :(return nothing),
    )
end

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
end

new_tester(x) = Foo(x)

type_unstable_tester(x::Ref{Any}) = cos(x[])

function pi_node_tester(y::Ref{Any})
    x = y[]
    return isa(x, Int) ? sin(x) : cos(x)
end

function avoid_throwing_path_tester(x)
    if x < 0
        Base.throw_boundserror(1:5, 6)
    end
    return sin(x)
end

# Performance notes:
# 1. as ever, intrinsics require real care.
# 2. passing only pointers, and loading / storing, seems to solve performance problems.
# 3. OpaqueClosures seem to work really quite well on 1.10.
# 4. I need to think carefully about how to ensure that DynamicDispatch works properly. At
#   present, I'm looking stuff up based on the static type of variables, however, I need the
#   dynamic type in order to do dispatch in the general case. This just means deferring the
#   construction of `InterpretedFunction`s until runtime in the case that the line is not
#   type stable. Simply use `isconcretetype` to check.
# 5. Need special handling for throw -- it doesn't play nicely with my closures.
#   Alterntaively, I could construct the OpaqueClosures lazily.
