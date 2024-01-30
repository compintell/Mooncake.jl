# Special types to represent data in an IRCode and a InterpretedFunction.

abstract type AbstractSlot{T} end

"""
    SlotRef{T}()

Constructs a reference to a slot of type `T` whose value is unassigned.

    SlotRef(x::T) where {T}

Constructs a reference to a slot of type `T` whose value is `x`.

    SlotRef{T}(x)

Constructs a reference to a slot of type `T` whose value is `x`. Valid provided that
`typeof(x) <: T`.
"""
mutable struct SlotRef{T} <: AbstractSlot{T}
    x::T
    SlotRef{T}() where {T} = new{T}()
    SlotRef(x::T) where {T} = new{T}(x)
    SlotRef{T}(x) where {T} = new{T}(x)
end

Base.getindex(x::SlotRef) = getfield(x, :x)
Base.setindex!(x::SlotRef, val) = setfield!(x, :x, val)
Base.isassigned(x::SlotRef) = isdefined(x, :x)
Base.eltype(::SlotRef{T}) where {T} = T
Base.copy(x::SlotRef{T}) where {T} = isassigned(x) ? SlotRef{T}(x[]) : SlotRef{T}()

"""
    ConstSlot(x)

Represents a constant, and it type-stable, and is therefore stored inline.
"""
struct ConstSlot{T} <: AbstractSlot{T}
    x::T
    ConstSlot(x::T) where {T} = new{T}(x)
    ConstSlot(::Type{T}) where {T} = new{Type{T}}(T)
    ConstSlot{T}(x) where {T} = new{T}(x)
end

Base.getindex(x::ConstSlot) = getfield(x, :x)
Base.setindex!(::ConstSlot, val) = nothing
Base.isassigned(::ConstSlot) = true
Base.eltype(::ConstSlot{T}) where {T} = T
Base.copy(x::ConstSlot{T}) where {T} = ConstSlot{T}(x[])

"""
    TypedGlobalRef(x::GlobalRef)

A (potentially) type-stable getter for a `GlobalRef`. In particular, if the `GlobalRef` is
declared to be a concrete type, this `getindex(::TypedGlobalRef)` will be type-stable. If no
declaration was made, then `getindex(::TypedGlobalRef)` will infer to `Any`.

If a `GlobalRef` is declared to be constant, prefer to represent it using a `ConstSlot`,
rather than a `TypedGlobalRef`.
"""
struct TypedGlobalRef{T} <: AbstractSlot{T}
    mod::Module
    name::Symbol
    TypedGlobalRef(x::GlobalRef) = new{x.binding.ty}(x.mod, x.name)
end

TypedGlobalRef(mod::Module, name::Symbol) = TypedGlobalRef(GlobalRef(mod, name))

Base.getindex(x::TypedGlobalRef{T}) where {T} = getglobal(x.mod, x.name)::T
Base.setindex!(x::TypedGlobalRef, val) = setglobal!(x.mod, x.name, val)
Base.isassigned(::TypedGlobalRef) = true
Base.eltype(::TypedGlobalRef{T}) where {T} = T

#=
Returns either a `ConstSlot` or a `TypedGlobalRef`, both of which are `AbstractSlot`s.
In particular, a `ConstSlot` is returned only if the `ex` is declared to be constant.
=#
function _globalref_to_slot(ex::GlobalRef)
    return isconst(ex) ? ConstSlot(getglobal(ex.mod, ex.name)) : TypedGlobalRef(ex)
end

# Utility functionality used through instruction construction.

const Inst = Core.OpaqueClosure{Tuple{Int}, Int}

# Standard handling for next-block returns for non control flow related instructions.
_standard_next_block(is_blk_end::Bool, current_blk::Int) = is_blk_end ? current_blk + 1 : 0


# IR node handlers -- translates Julia SSAIR nodes into executable `Inst`s (see above).
# Each node may have several methods of `build_inst`. One will always be a method which
# accepts a variety of arguments, including an `InterpretedFunction`, extracts only the data
# that it needs, and calls another method of `build_inst`. This other method of
# `build_inst` will actually build the instruction. This structure is used to make it
# easy to construct unit test cases for the second method (if you inspect the tests for this
# code, you will find that the second method is usually called).

## ReturnNode
function build_inst(inst::ReturnNode, @nospecialize(in_f), ::Int, ::Int, ::Bool)::Inst
    return build_inst(ReturnNode, in_f.return_slot, _get_slot(inst.val, in_f))
end
function build_inst(::Type{ReturnNode}, ret_slot::SlotRef, val_slot::AbstractSlot)
    return @opaque (prev_block::Int) -> (ret_slot[] = val_slot[]; return -1)
end

## GotoNode
function build_inst(inst::GotoNode, @nospecialize(in_f), ::Int, ::Int, ::Bool)::Inst
    return build_inst(GotoNode, inst.label)
end
build_inst(::Type{GotoNode}, label::Int) = @opaque (p::Int) -> label

## GotoIfNot
function build_inst(x::GotoIfNot, @nospecialize(in_f), ::Int, b::Int, ::Bool)::Inst
    return build_inst(GotoIfNot, _get_slot(x.cond, in_f), b + 1, x.dest)
end
function build_inst(::Type{GotoIfNot}, cond::AbstractSlot, next_blk::Int, dest::Int)
    if !(Bool <: eltype(cond))
        throw(ArgumentError("cond $cond has eltype $(eltype(cond)), not a supertype of Bool"))
    end
    return @opaque (p::Int) -> cond[] ? next_blk : dest
end

## PhiNode

struct TypedPhiNode{Tr<:AbstractSlot, Te<:Tuple, Tv<:Tuple}
    tmp_slot::Tr
    ret_slot::Tr
    edges::Te
    values::Tv
end

# Runs a collection of PhiNodes (semantically) simulataneously. Does this by first writing
# the value associated to each PhiNode to its `tmp_slot`. Once all values have been written,
# copies the `tmp_slot` value across to the `ret_slot`. This ensures that if e.g.
# PhiNode B takes the value associated to PhiNode A, it gets the value _before_ this
# collection of PhiNodes started to run, rather than after. See SSAIR docs for more info.
function build_phinode_insts(
    ir_insts::Vector{PhiNode}, in_f, n_first::Int, b::Int, is_blk_end::Bool
)::Inst
    nodes = build_typed_phi_nodes(ir_insts, in_f, n_first)
    next_blk = _standard_next_block(is_blk_end, b)
    return build_inst(Vector{PhiNode}, (nodes..., ), next_blk)
end

struct UndefRef end

function build_typed_phi_nodes(ir_insts::Vector{PhiNode}, in_f, n_first::Int)
    return map(enumerate(ir_insts)) do (j, ir_inst)
        ret_slot = in_f.slots[n_first + j - 1]
        edges = map(Int, (ir_inst.edges..., ))
        vals = ir_inst.values
        _init = map(eachindex(vals)) do j
            return isassigned(vals, j) ? _get_slot(vals[j], in_f) : UndefRef()
        end
        T = eltype(ret_slot)
        values_vec = map(n -> _init[n] isa UndefRef ? SlotRef{T}() : _init[n], eachindex(_init))
        return TypedPhiNode(copy(ret_slot), ret_slot, edges, (values_vec..., ))
    end
end

function build_inst(::Type{Vector{PhiNode}}, nodes::Tuple, next_blk::Int)::Inst
    return @opaque function (prev_blk::Int)
        map(Base.Fix2(store_tmp_value!, prev_blk), nodes)
        map(transfer_tmp_value!, nodes)
        return next_blk
    end
end

function store_tmp_value!(node::TypedPhiNode, prev_blk::Int)
    map(node.edges, node.values) do edge, val
        (edge == prev_blk) && isassigned(val) && (node.tmp_slot[] = val[])
    end
    return nothing
end

function transfer_tmp_value!(node::TypedPhiNode)
    isassigned(node.tmp_slot) && (node.ret_slot[] = node.tmp_slot[])
    return nothing
end

## PiNode
function build_inst(x::PiNode, @nospecialize(in_f), n::Int, b::Int, is_blk_end::Bool)::Inst
    next_blk = _standard_next_block(is_blk_end, b)
    return build_inst(PiNode, _get_slot(x.val, in_f), in_f.slots[n], next_blk)
end
function build_inst(::Type{PiNode}, input::AbstractSlot, out::AbstractSlot, next_blk::Int)
    return @opaque (prev_blk::Int) -> (out[] = input[]; return next_blk)
end

## GlobalRef
function build_inst(x::GlobalRef, @nospecialize(in_f), n::Int, b::Int, is_blk_end::Bool)::Inst
    next_blk = _standard_next_block(is_blk_end, b)
    return build_inst(GlobalRef, _globalref_to_slot(x), in_f.slots[n], next_blk)
end
function build_inst(::Type{GlobalRef}, x::AbstractSlot, out::AbstractSlot, next_blk::Int)
    return @opaque (prev_blk::Int) -> (out[] = x[]; return next_blk)
end

## QuoteNode and literals
function build_inst(node, @nospecialize(in_f), n::Int, b::Int, is_blk_end::Bool)::Inst
    x = ConstSlot(node isa QuoteNode ? node.value : node)
    return build_inst(nothing, x, in_f.slots[n], _standard_next_block(is_blk_end, b))
end
function build_inst(::Nothing, x::ConstSlot, out_slot::AbstractSlot, next_blk::Int)
    return @opaque (prev_blk::Int) -> (out_slot[] = x[]; return next_blk)
end

## Expr

@inline _eval(f::F, args::Vararg{Any, N}) where {F, N} = f(args...)

tangent_type(::Type{typeof(_eval)}) = NoTangent

function build_inst(x::Expr, @nospecialize(in_f), n::Int, b::Int, is_blk_end::Bool)::Inst
    next_blk = _standard_next_block(is_blk_end, b)
    val_slot = in_f.slots[n]
    if Meta.isexpr(x, :boundscheck)
        return build_inst(Val(:boundscheck), val_slot, next_blk)
    elseif Meta.isexpr(x, :invoke) || Meta.isexpr(x, :call)
        is_invoke = Meta.isexpr(x, :invoke)
        __args = is_invoke ? x.args[2:end] : x.args
        arg_refs = map(arg -> _get_slot(arg, in_f), (__args..., ))
        sig = Tuple{map(eltype, arg_refs)...}
        evaluator = get_evaluator(in_f.ctx, sig, in_f.interp, is_invoke)
        return build_inst(Val(:call), arg_refs, evaluator, val_slot, next_blk)
    elseif x.head in [
        :code_coverage_effect, :gc_preserve_begin, :gc_preserve_end, :loopinfo, :leave,
        :pop_exception,
    ]
        return build_inst(Val(:skipped_expression), next_blk)
    elseif Meta.isexpr(x, :throw_undef_if_not)
        slot_to_check = _get_slot(x.args[2], in_f)
        return build_inst(Val(:throw_undef_if_not), slot_to_check, next_blk)
    else
        throw(error("Unrecognised expression $x"))
    end
end

function get_evaluator(ctx::T, sig, interp, is_invoke::Bool) where {T}
    is_primitive(ctx, sig) && return _eval
    is_invoke && return InterpretedFunction(ctx, sig, interp)
    return DelayedInterpretedFunction(ctx, Dict(), interp)
end

function build_inst(::Val{:boundscheck}, val_slot::AbstractSlot, next_blk::Int)::Inst
    return @opaque (prev_blk::Int) -> (val_slot[] = true; return next_blk)
end

function build_inst(
    ::Val{:call},
    arg_slots::Targ_slots,
    ev::Teval,
    val_slot::AbstractSlot,
    next_blk::Int,
)::Inst where {Teval, Targ_slots}
    inst = @opaque function (prev_blk::Int)
        val_slot[] = ev(tuple_map(getindex, arg_slots)...)
        return next_blk
    end
    return inst
end

build_inst(::Val{:skipped_expression}, next_blk::Int)::Inst = @opaque (prev_blk::Int) -> next_blk

function build_inst(::Val{:throw_undef_if_not}, slot_to_check::AbstractSlot, next_blk::Int)::Inst
    return @opaque function (prev_blk::Int)
        !isassigned(slot_to_check) && throw(error("Boooo, not assigned"))
        return next_blk
    end
end

#
# Code execution
#

_get_slot(x::Argument, _, arg_info, _) = arg_info.arg_slots[x.n]
_get_slot(x::GlobalRef, _, _, _) = _globalref_to_slot(x)
_get_slot(x::QuoteNode, _, _, _) = ConstSlot(x.value)
_get_slot(x::SSAValue, slots, _, _) = slots[x.id]
_get_slot(x::AbstractSlot, _, _, _) = throw(error("Already a slot!"))
_get_slot(x, _, _, _) = ConstSlot(x)
function _get_slot(x::Expr, _, _, sptypes)
    # There are only a couple of `Expr`s possible as arguments to `Expr`s.
    if Meta.isexpr(x, :boundscheck)
        return ConstSlot(true)
    elseif Meta.isexpr(x, :static_parameter)
        return ConstSlot(sptypes[x.args[1]].typ)
    else
        throw(ArgumentError("Found unexpected expr $x"))
    end
end



#
# Loading arguments into slots.
#

# Data structure to handle arguments to functions. Comprises a collection of slots, and
# knows whether or not it represents the arguments of a varargs function.
struct ArgInfo{Targ_slots<:NTuple{N, Any} where {N}, is_vararg}
    arg_slots::Targ_slots
end

function ArgInfo(::Type{T}, is_vararg::Bool) where {T<:Tuple}
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
        num_args = length(ai.arg_slots) - 1 # once for vararg
        refined_args = (args[1:num_args]..., (args[num_args+1:end]..., ))
    else
        refined_args = args
    end
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
    slots::Vector{AbstractSlot}
    instructions::Vector{Inst}
    bb_starts::Vector{Int}
    bb_ends::Vector{Int}
    ir::IRCode
    interp::TapedInterpreter
    spnames::Any
end

# See above for other `_get_slot` methods.
function _get_slot(x, in_f::InterpretedFunction)
    return _get_slot(x, in_f.slots, in_f.arg_info, in_f.ir.sptypes)
end

function is_vararg_sig_and_sparam_names(sig)
    world = Base.get_world_counter()
    min = Base.RefValue{UInt}(typemin(UInt))
    max = Base.RefValue{UInt}(typemax(UInt))
    ms = Base._methods_by_ftype(sig, nothing, -1, world, true, min, max, Ptr{Int32}(C_NULL))::Vector
    m = only(ms).method
    return m.isva, sparam_names(m)
end

function sparam_names(m::Core.Method)::Vector{Symbol}
    whereparams = ExprTools.where_parameters(m.sig)
    whereparams === nothing && return Symbol[]
    return map(whereparams) do name
        name isa Symbol && return name
        Meta.isexpr(name, :(<:)) && return name.args[1]
        Meta.isexpr(name, :(>:)) && return name.args[1]
        error("unrecognised type param $name")
    end
end

make_slot(x::Type{T}) where {T} = (@isdefined T) ? SlotRef{T}() : SlotRef{DataType}()
make_slot(x::CC.Const) = ConstSlot{Core.Typeof(x.val)}(x.val)
make_slot(x::CC.PartialStruct) = SlotRef{x.typ}()
make_slot(::CC.PartialTypeVar) = SlotRef{TypeVar}()

make_dummy_instruction(next_blk::Int) = @opaque (p::Int) -> next_blk

# Special handling is required for PhiNodes, because their semantics require that when
# more than one PhiNode appears at the start of a basic block, they are run simulataneously
# rather than in sequence. See the SSAIR docs for an explanation of why this is the case.
function make_phi_instructions!(in_f::InterpretedFunction)
    ir = in_f.ir
    insts = in_f.instructions
    for (b, bb) in enumerate(ir.cfg.blocks)

        # Find any phi nodes at the start of the block.
        phi_node_inds = Int[]
        foreach(n -> (ir.stmts.inst[n] isa PhiNode) && push!(phi_node_inds, n), bb.stmts)
        isempty(phi_node_inds) && continue

        # Make a single instruction which runs all of the PhiNodes "simulataneously".
        # Specifically, this instruction runs all of the phi nodes, storing the results of
        # this into temporary storage, then writing from the temporary slots to the
        # final slots. This has the effect of ensuring that phi nodes that depend on other
        # phi nodes get the "old" values, not the new updated values. This was a
        # surprisingly hard bug to catch and resolve.
        nodes = [ir.stmts.inst[n] for n in phi_node_inds]
        n_first = first(phi_node_inds)
        is_blk_end = length(phi_node_inds) == length(bb.stmts)
        insts[phi_node_inds[1]] = build_phinode_insts(nodes, in_f, n_first, b, is_blk_end)

        # Create dummy instructions for the remainder of the nodes.
        for n in phi_node_inds[2:end]
            insts[n] = make_dummy_instruction(_standard_next_block(is_blk_end, b))
        end
    end
    return nothing
end

"""
    InterpretedFunction(ctx::C, sig::Type{<:Tuple}, interp) where {C}

Construct a data structure which can be used to execute the instruction specified by `sig`.
For example,
```julia
in_f = InterpretedFunction(DefaultCtx(), Tuple{typeof(sin), Float64}, Taped.TInterp())
in_f(sin, 5.0)
```
will yield exactly the same result as running `sin(5.0)`. The advantage of this data
structure is that `build_rrule!!` is implemented for it, meaning that it can be
differentiated.

The performance of `InterpretedFunction` largely depends on what the functions are that it
operates on, but it definitely adds a notable amount of overhead when compared to regular
Julia code. Typically this overhead is on the order 10ns per operation (on a modern CPU).

For example, running on low-level code involving small scalar operations will
_typically_ take 10-100 times longer than running the original Julia function, but BLAS
calls on moderately large matrices has negligible overhead when compared with the original
function.

## Caching

`InterpretedFunction`s are cached by `interp` -- as a consequence, if you call
`InterpretedFunction` twice with the same arguments, the second call will just return a
cached result.

## Known-Limitations

While much of the language is supported, there are a few things that `InterpretedFunction`
_cannot_ execute. These include anything to do with threading, and exception handling.
The ability to handling threading may improve in future versions of `InterpretedFunction`,
but exception handling is unlikely to be supported, as it is not at all clear how it would
be handled in reverse-mode AD.

Note that `InterpretedFunction`s should be fine with the constructs involved in exception
handling _provided_ that no exceptions are actually thrown.

# Implementation

An `InterpretedFunction` operates by first looking up the _optimised_ IRCode associated to
`sig` under `interp`. It associates each instruction in the IR with a `Core.OpaqueClosure`,
and each `Argument` / `SSAValue` in the IR with a (heap-allocated) `AbstractSlot` (for the
most part, these slots are `Ref`s).

While the details of what each kind of `OpaqueClosure` can be found in the corresponding
`Taped.build_inst` method, they generally have the following structure:
- load data from argument / ssa slots,
- do computation,
- write result to the instruction's ssa slot,
- return an integer indicating which instruction to execute next.

The returned integer is permitted to take one of the following values:
- `-1`, in which case we should return,
- `0`, in which case the next instruction should be run,
- a positive integer, in which case execution jumps to the start of that block.

The only argument to each `Core.OpaqueClosure` is an integer corresponding to the index of
the previous block that was run. As a result, each instruction has the _same_ signature,
meaning that while each instruction tends to do quite different things, we do not see an
explosion of types. Moreover, type-stability it maintained.
"""
function InterpretedFunction(ctx::C, sig::Type{<:Tuple}, interp) where {C}

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
    arg_info = ArgInfo(arg_types, is_vararg)
    ir = normalise!(ir, spnames)

    # Create slots. In most cases, these are instances of `SlotRef`s, which can be read from
    # and written to by instructions (they are essentially `Base.RefValue`s with a
    # different name. Very occassionally the compiler will deduce that a particular slot has
    # a constant value. In these cases, we instead create an instance of `ConstSlot`, which
    # cannot be written to.
    slots = AbstractSlot[make_slot(T) for T in ir.stmts.type]

    # Allocate memory for instructions and argument loading instructions.
    insts = Vector{Inst}(undef, length(slots))

    # Compute the index of the instruction associated with the start of each basic block
    # in `ir`. This is used to know where to jump to when we hit a `Core.GotoNode` or
    # `Core.GotoIfNot`. The `ir.cfg` very nearly gives this to us for free.
    bb_starts = vcat(1, ir.cfg.index)
    bb_ends = vcat(ir.cfg.index .- 1, length(slots))

    # Extract the starting location of each basic block from the CFG and build IF.
    in_f = InterpretedFunction{sig, C, Treturn, Core.Typeof(arg_info)}(
        ctx, return_slot, arg_info, slots, insts, bb_starts, bb_ends, ir, interp, spnames,
    )

    # Eagerly create PhiNode instructions, as this requires special handling.
    make_phi_instructions!(in_f)

    # Cache this InterpretedFunction so that we don't have tobuild it again.
    interp.in_f_cache[sig] = in_f
    return in_f
end

function (in_f::InterpretedFunction)(args::Vararg{Any, N}) where {N}
    load_args!(in_f, args)
    return __barrier(in_f)
end

load_args!(in_f::InterpretedFunction, args::Targs) where {Targs} = load_args!(in_f.arg_info, args)

# Execute an interpreted function, having already loaded the arguments into their slots.
function __barrier(in_f::Tf) where {Tf<:InterpretedFunction}
    prev_block = 0
    next_block = 0
    current_block = 1
    n = 1
    instructions = in_f.instructions
    while next_block != -1
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

"""
    tuple_map(f::F, x::Tuple) where {F}

This function is semantically equivalent to `map(f, x)`, but always specialises on all of
the element types of `x`, regardless the length of `x`. This contrasts with `map`, in which
the number of element types specialised upon is a fixed constant in the compiler.

As a consequence, if `x` is very long, this function may have very large compile times.
"""
@generated function tuple_map(f::F, x::Tuple) where {F}
    return Expr(:call, :tuple, map(n -> :(f(x[$n])), eachindex(x.parameters))...)
end

# Produce a `Dict` mapping from block numbers to line number of their first statement.
function block_map(cfg::CC.CFG)
    line_to_blk_maps = map(((n, blk),) -> tuple.(blk.stmts, n), enumerate(cfg.blocks))
    return Dict(reduce(vcat, line_to_blk_maps))
end

function build_inst(in_f::InterpretedFunction{sig}, n::Int) where {sig}
    @nospecialize in_f
    ir_inst = in_f.ir.stmts.inst[n]
    b = block_map(in_f.ir.cfg)[n]
    is_blk_end = n in in_f.bb_ends
    return build_inst(ir_inst, in_f, n, b, is_blk_end)
end

# Use to handle dynamic dispatch inside `InterpretedFunction`s.
#
# `InterpretedFunction`s operate recursively -- if the types associated to the `args` field
# of a `:call` expression have not been inferred successfully, then we must wait until
# runtime to determine what code to run. The `DelayedInterpretedFunction` does exactly this.
struct DelayedInterpretedFunction{C, Tlocal_cache, T<:TapedInterpreter}
    ctx::C
    local_cache::Tlocal_cache
    interp::T
end

compute_oc_type(::Type{sig}) where {sig<:Tuple} = OpaqueClosure{sig, CC.return_type(sig)}

@generated function (din_f::DelayedInterpretedFunction)(fargs::Vararg{Any, N}) where {N}
    sig = Tuple{fargs...}
    sig_id = objectid(sig)
    oc_type = compute_oc_type(sig)
    return quote
        local_cache = din_f.local_cache
        _evaluator = if in($sig_id, keys(local_cache))
            local_cache[$sig_id]::$oc_type
        else
            derive_function!(din_f.ctx, local_cache, din_f.interp, $sig)::$oc_type
        end
        return _evaluator(fargs...)
    end
end

function derive_function!(ctx, local_cache, interp, sig)
    _evaluator = if is_primitive(ctx, sig)
        _eval
    else
        InterpretedFunction(ctx, sig, interp)
    end
    oc_eval = @opaque sig (x...) -> _evaluator(x...)
    local_cache[objectid(sig)] = oc_eval
    return oc_eval
end
