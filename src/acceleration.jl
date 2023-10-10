# The approach taken to "accelerating" the tape in this code is precisely the same one taken
# by ReverseDiff.jl, albeit the approach taken here is not AD-specific -- it can be applied
# to any Umlaut.jl tape.

struct ConstArg{T}
    x::T
end

@inline dereference(x::ConstArg) = x.x
@inline dereference(x::Ref) = x[]

@inline assign!(::ConstArg, y) = nothing
@inline function assign!(x::Ref, y)
    x[] = y
    return nothing
end

struct Instruction{Tf, Targ_refs, Tval_ref}
    f::Tf
    arg_refs::Targ_refs
    val_ref::Tval_ref
end

@noinline function _forward_exec!(inst::Instruction)
    args = map(dereference, inst.arg_refs)
    val = inst.f(args...)
    assign!(inst.val_ref, val)
    return nothing
end

@noinline _execute!(inst::Instruction) = _forward_exec!(inst)

struct Executor{T}
    inst::T
end

@inline (e::Executor)() = _execute!(e.inst)

function wrapped_instruction(f, arg_refs, val_ref)
    return FunctionWrapper{Nothing, Tuple{}}(Executor(Instruction(f, arg_refs, val_ref)))
end

struct AcceleratedTape{Targ_refs, Tval_ref}
    instructions::Vector{FunctionWrapper{Nothing, Tuple{}}}
    arg_refs::Targ_refs
    val_ref::Tval_ref
end

function execute!(tape::AcceleratedTape{T}, args::Vararg{Any, N}) where {T, N}
    set_arg_vals!(tape, args)
    for inst in tape.instructions
        inst()
    end
    return get_return_val(tape)
end

@noinline set_arg_vals!(tape::AcceleratedTape, args) = map(assign!, tape.arg_refs, args)

get_return_val(tape::AcceleratedTape) = tape.val_ref[]

num_inputs(tape::Tape) = length(inputs(tape))

make_ref(x::T) where {T} = Base.issingletontype(T) ? ConstArg(x) : Ref(x)

function accelerate(tape::Tape)
    slot_refs = make_ref.(getproperty.(tape.ops, :val))
    instructions = create_instruction.(Ref(slot_refs), tape.ops)
    arg_refs = (slot_refs[1:num_inputs(tape)]..., )
    val_ref = slot_refs[tape.result.id]
    return AcceleratedTape(instructions, arg_refs, val_ref)
end

no_op() = nothing

create_instruction(_, ::Constant) = wrapped_instruction(no_op, (), Ref(nothing))
create_instruction(_, ::Input) = wrapped_instruction(no_op, (), Ref(nothing))
function create_instruction(slot_refs, op::Call)
    arg_refs = (map(arg -> get_ref(slot_refs, arg), op.args)..., )
    val_ref = get_ref(slot_refs, op)
    return wrapped_instruction(op.fn, arg_refs, val_ref)
end

get_ref(slot_refs, arg::Union{Umlaut.AbstractOp, Variable}) = slot_refs[arg.id]
get_ref(_, arg) = ConstArg(arg)
