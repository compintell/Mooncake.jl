#
# Special Ref type to avoid confusion between existing ref types.
#

mutable struct SlotRef{T}
    x::T
    SlotRef{T}() where {T} = new{T}()
    SlotRef(x::T) where {T} = new{T}(x)
end

Base.getindex(x::SlotRef) = x.x
function Base.setindex!(x::SlotRef, val)
    setfield!(x, :x, val)
    return x.x
end

extract_arg(x::SlotRef{T}) where {T} = x[]
extract_arg(x::T) where {T} = x # assume literal



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
        :(args = inst.args),
        Expr(
            :(=),
            :(inst.val_ref[]),
            Expr(:new, T, map(n -> :(extract_arg(args[$n])), 1:nargs)...),
        ),
        :(return 0),
    )
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
#

struct PhiNodeInst{Tedges<:Tuple, Tvalues<:Tuple, Tval_slot<:SlotRef}
    edges::Tedges
    values::Tvalues
    val_slot::Tval_slot
end

function (inst::PhiNodeInst)(prev_block::Int, ::Int)
    # display(inst.values)
    # println()
    # vals = map(extract_arg, inst.values) # extract all for type stability
    for n in eachindex(inst.edges)
        if inst.edges[n] == prev_block
            inst.val_slot[] = extract_arg(inst.values[n])#vals[n]
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

struct SkippedExpressionInst
    s::Symbol
end

(::SkippedExpressionInst)(::Int, ::Int) = 0

#
# Code execution
#

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

const IFInstruction = Core.OpaqueClosure{Tuple{Int, Int}, Int}

function _make_opaque_closure(inst, sig, n)
    oc = Base.Experimental.@opaque Tuple{Int, Int} (p, q) -> inst(p, q)
    if !(oc isa IFInstruction)
        println("Displaying debugging info from _make_opaque_closure:")
        println("sig of InterpretedFunction in which this is instruction $n:")
        display(sig)
        println()
        println("inst:")
        display(inst)
        println()
        println("IRCode from inst (source of failure) has argtypes:")
        display(new_ir.argtypes)
        println()
        println("and IR")
        display(new_ir)
        println()
    end
    return oc::IFInstruction
end

get_tuple_type(x::Tuple) = Tuple{map(_other_robust_typeof, x)...}

_get_arg_type(::Type{Val{T}}) where {T} = T

"""
    function _foreigncall_(
        ::Val{name}, ::Val{RT}, AT::Tuple, ::Val{nreq}, ::Val{calling_convention}, x...
    ) where {name, RT, nreq, calling_convention}

:foreigncall nodes get translated into calls to this function.
For example,
```julia
Expr(:foreigncall, :foo, Tout, (A, B), nreq, :ccall, args...)
```
becomes
```julia
_foreigncall_(Val(:foo), Val(Tout), (Val(A), Val(B)), Val(nreq), Val(:ccall), args...)
```
Please consult the Julia documentation for more information on how foreigncall nodes work,
and consult this package's tests for examples.

Credit: Umlaut.jl has the original implementation of this function. This is largely copied
over from there.
"""
@generated function _foreigncall_(
    ::Val{name}, ::Val{RT}, AT::Tuple, ::Val{nreq}, ::Val{calling_convention}, x...
) where {name, RT, nreq, calling_convention}
    return Expr(
        :foreigncall,
        QuoteNode(name),
        :($(RT)),
        Expr(:call, :(Core.svec), map(_get_arg_type, AT.parameters)...),
        :($nreq),
        QuoteNode(calling_convention),
        map(n -> :(x[$n]), 1:length(x))...,
    )
end

is_primitive(::Type{<:Tuple{typeof(_foreigncall_), Vararg{Any, N}}} where {N}) = true

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
    st = Core.Compiler.copy(st)
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
    return Meta.isexpr(st, :(=)) ? Expr(:(=), st.args[1], ex) : ex
end
rewrite_special_cases(::IRCode, st) = st
function rewrite_special_cases(ir::IRCode, st::Core.GotoIfNot)
    return Core.GotoIfNot(rewrite_special_cases(ir, st.cond), st.dest)
end

struct InterpretedFunction{sig, Treturn_slot<:SlotRef, Targ_slots, Tslots}
    return_slot::Treturn_slot
    arg_slots::Targ_slots
    slots::Tslots
    instructions::Vector{IFInstruction}
    bb_starts::Vector{Int}
    ir::IRCode
end

# const __IF_METHOD_TABLE = Dict{Type{<:Tuple}, InterpretedFunction}()

# _get_sig(::InterpretedFunction{sig, Tr, Ta, Ts}) where {sig, Tr, Ta, Ts} = Tr, Ta, Ts

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
    instructions = Vector{IFInstruction}(undef, length(slots))

    # Extract the starting location of each basic block from the CFG and build IF.
    return InterpretedFunction{sig, Treturn_slot, Targ_slots, Tslots}(
        return_slot, arg_slots, slots, instructions, vcat(1, ir.cfg.index), ir
    )
end

function (in_f::InterpretedFunction)(args::Vararg{Any, N}) where {N}
    load_args!(in_f, args)
    prev_block = 0
    next_block = 0
    current_block = 1
    n = 1
    instructions = in_f.instructions
    while next_block != -1
        # @show prev_block, next_block, current_block, n
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

@generated function load_args!(in_f, args::Tuple)
    return Expr(
        :block,
        map(n -> :(in_f.arg_slots[$(n + 1)][] = args[$n]), eachindex(args.parameters))...,
        :(return nothing),
    )
end

function build_instruction(in_f::InterpretedFunction{sig}, n::Int) where {sig}
    ir = in_f.ir
    ir_inst = rewrite_special_cases(ir, ir.stmts.inst[n])
    is_invoke = Meta.isexpr(ir_inst, :invoke)
    arg_slots = in_f.arg_slots
    slots = in_f.slots
    return_slot = in_f.return_slot
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

        fn_sig = Tuple{Core.Typeof(fn), map(_robust_typeof, arg_refs)...}
        if !is_primitive(fn_sig)
            if length(Base.code_ircode_by_type(fn_sig)) == 1
                fn = InterpretedFunction(fn_sig)
            else
                fn = DelayedInterpretedFunction{Core.Typeof(fn)}(fn)
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
        values = map(x -> x isa SlotRef ? x : SlotRef(x), (values_vec..., ))
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
    elseif ir_inst isa Expr && ir_inst.head in [
        :code_coverage_effect, :gc_preserve_begin, :gc_preserve_end, :loopinfo,
        :leave, :pop_exception,
    ]
        SkippedExpressionInst(ir_inst.head)
    else
        println("IR in which error is found:")
        display(sig)
        display(ir)
        println()
        throw(error("unhandled instruction $ir_inst, with type $(typeof(ir_inst))")) 
    end
    return _make_opaque_closure(inst, sig, n)
end

struct DelayedInterpretedFunction{F}
    f::F
end

function (f::DelayedInterpretedFunction{F})(args...) where {F}
    sig = Tuple{F, Core.Typeof(args).parameters...}
    return is_primitive(sig) ? f.f(args...) : InterpretedFunction(sig)(args...)
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

function foreigncall_tester(x)
    X = @ccall jl_alloc_array_2d(Matrix{Float64}::Any, 5::Int, 5::Int)::Matrix{Float64}
    for n in eachindex(X)
        X[n] = x
    end
    return X
end

# TODO notes:
# 3. OpaqueClosures seem to work really quite well on 1.10.
# 4. I need to think carefully about how to ensure that DynamicDispatch works properly. At
#   present, I'm looking stuff up based on the static type of variables, however, I need the
#   dynamic type in order to do dispatch in the general case. This just means deferring the
#   construction of `InterpretedFunction`s until runtime in the case that the line is not
#   type stable. Simply use `isconcretetype` to check.
