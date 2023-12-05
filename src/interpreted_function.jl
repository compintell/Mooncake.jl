struct InterpretedFunction{sig, Treturn_slot<:Ref, Targ_slots, Tslots}
    return_slot::Treturn_slot
    arg_slots::Targ_slots
    slots::Tslots
    instructions::Vector{FunctionWrapper{Int, Tuple{Int}}}
end

_type(x) = x
_type(x::Core.Const) = Core.Typeof(x.val)

struct RunAndIncrement{Tinst}
    inst::Tinst
end

function (f::RunAndIncrement{A})(n) where {A}
    f.inst()
    return n + 1
end

struct PtrInstruction{Tf, Targ_ptrs, Tval_ptr<:Ptr}
    f::Tf
    arg_ptrs::Targ_ptrs
    val_ptr::Tval_ptr
end

function (inst::PtrInstruction{A, B, C})(n::Int) where {A, B, C}
    arg_vals = map(unsafe_load, inst.arg_ptrs)
    val = inst.f(arg_vals...)
    unsafe_store!(inst.val_ptr, val)
    return n + 1
end

struct ReturnInstruction{Treturn_slot, Tval_slot}
    return_slot::Treturn_slot
    val_slot::Tval_slot
end

# Returning -1 indicates that the calling function should return.
function (inst::ReturnInstruction{A, B})(n::Int) where {A, B}
    unsafe_store!(inst.return_slot, unsafe_load(inst.val_slot))
    return -1
end

get_pointer(x::Ref{T}) where {T} = convert(Ptr{T}, pointer_from_objref(x))

function InterpretedFunction(sig::Type{<:Tuple})

    # Grab code associated to this function.
    ir, Treturn = Base.code_ircode_by_type(sig)[1]

    # Construct return slot.
    return_slot = Ref{Treturn}()
    Treturn_slot = Core.Typeof(return_slot)

    # Construct argument reference references.
    arg_slots = map(T -> Ref{_type(T)}(), (ir.argtypes..., ))
    Targ_slots = Core.Typeof(arg_slots)

    # Extract slot types.
    slot_types = tuple(ir.stmts.type...)
    slots = map(T -> Ref{_type(T)}(), slot_types)
    Tslots = Core.Typeof(slots)

    # Construct instructions (we should ideally do this recursively, but can't yet).
    instructions = map(eachindex(slots)) do n
        ir_inst = ir.stmts.inst[n]
        if Meta.isexpr(ir_inst, :invoke)
            fn = ir_inst.args[2]
            if fn isa GlobalRef
                fn = getglobal(fn.mod, fn.name)
            end

            _arg_ptrs = map((ir_inst.args[3:end]..., )) do arg
                if arg isa Core.SSAValue
                    return get_pointer(slots[arg.id])
                elseif arg isa Core.Argument
                    return get_pointer(arg_slots[arg.n])
                else
                    throw(error("boo"))
                end
            end

            val_ptr = get_pointer(slots[n])

            return PtrInstruction(fn, _arg_ptrs, val_ptr)
        elseif ir_inst isa Core.ReturnNode
            v = ir_inst.val
            arg_slot = v isa Core.SSAValue ? slots[v.id] : arg_slots[v.n]
            return ReturnInstruction(get_pointer(return_slot), get_pointer(arg_slot))
        else
            throw(error("unhandled instruction $inst")) 
        end
    end

    return InterpretedFunction{sig, Treturn_slot, Targ_slots, Tslots}(
        return_slot, arg_slots, slots, instructions
    )
end

function (in_f::InterpretedFunction)(args...)

    # Load argument values into argument refs.
    foreach(zip(in_f.arg_slots, args)) do (arg_ref, arg)
        arg_ref[] = arg
    end

    # Run code until told to return.
    n_prev = 1
    n = 1
    insts = in_f.instructions
    while n != -1
        n_prev = n
        n = insts[n](n_prev)
    end

    return in_f.return_slot[]
end
