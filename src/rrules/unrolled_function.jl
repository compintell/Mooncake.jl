using Base: RefValue

struct CoInstruction{Tinputs<:Tuple{Vararg{RefValue}}, Toutput<:RefValue, Tpb}
    inputs::Tinputs
    output::Toutput
    pb::Tpb
end

const_coinstruction(x::CoDual) = CoInstruction((), Ref(x), nothing)

input_primals(x::CoInstruction) = map(primal ∘ getindex, x.inputs)
input_tangents(x::CoInstruction) = map(tangent ∘ getindex, x.inputs)

output_primal(x::CoInstruction) = primal(x.output[])
output_tangent(x::CoInstruction) = tangent(x.output[])

function seed_output_tangent!(x::CoInstruction{T, V}, x̄) where {T, V}
    x.output[] = set_tangent!!(x.output[], x̄)
    return nothing
end

function optimised_rrule!!(args::Vararg{Any, N}) where {N}
    primals = map(primal, args)
    might_be_active(primals) && return rrule!!(args...)
    y = primals[1](primals[2:end]...)
    return CoDual(y, uninit_tangent(y)), NoPullback()
end

function build_coinstruction(inputs::CoInstruction...)
    input_refs = map(x -> x.output, inputs)
    input_values = map(getindex, input_refs)
    output_value, pb!! = optimised_rrule!!(input_values...)
    output_ref = Ref(output_value)
    pb_ref = Ref(pb!!)
    return CoInstruction(input_refs, output_ref, pb_ref)
end

function (instruction::CoInstruction)(inputs::Vararg{CoInstruction, N}) where {N}
    input_refs = map(x -> x.output, inputs)
    input_values = map(getindex, input_refs)
    foreach(verify_codual_type, input_values)
    output_value, pb!! = optimised_rrule!!(input_values...)
    verify_codual_type(output_value)
    output_ref = instruction.output
    output_ref[] = output_value
    pb_ref = instruction.pb
    pb_ref[] = pb!!
    return CoInstruction(input_refs, output_ref, pb_ref)
end

# pullback for "constant" CoInstruction.
pullback!(::CoInstruction{Tuple{}, <:Ref, Nothing}) = nothing

# pullback for general case CoInstruction.
function pullback!(instruction::CoInstruction)
    input_tangents = map(tangent ∘ getindex, instruction.inputs)
    output_tangent = tangent(instruction.output[])
    new_input_tangents = instruction.pb[](output_tangent, input_tangents...)
    map(replace_tangent!, instruction.inputs, new_input_tangents)
    return nothing
end

function replace_tangent!(x::Ref{<:CoDual{Tx, Tdx}}, new_tangent::Tdx) where {Tx, Tdx}
    x_val = x[]
    x[] = CoDual(primal(x_val), new_tangent)
    return nothing
end

function to_reverse_mode_ad(tape::Tape{RMC}, ȳ, inputs::CoInstruction...)
    inputs!(tape, inputs...)

    new_tape = Tape(tape.c)

    # Transform forwards pass, replacing ops with associated rrule calls.
    for op in tape.ops
        push!(new_tape, to_reverse_mode_ad(op, new_tape))
    end
    new_tape.result = unbind(tape.result)

    # Seed reverse-pass and create operations to execute it.
    seed_op = mkcall(seed_output_tangent!, new_tape.result, ȳ)
    push!(new_tape, seed_op)

    Umlaut.exec!(new_tape, seed_op)
    for op in reverse(new_tape.ops[1:end-1])
        pb_op = mkcall(pullback!, Variable(op.id))
        push!(new_tape, pb_op)
        Umlaut.exec!(new_tape, pb_op)
    end

    return new_tape
end

is_umlaut_type(x::Union{Variable, Constant, Input}) = true
is_umlaut_type(x) = false

to_reverse_mode_ad(x::Input, new_tape) = Input(x.val)
function to_reverse_mode_ad(x::Constant, new_tape)
    return Constant(const_coinstruction(CoDual(x.val, uninit_tangent(x.val))))
end
function to_reverse_mode_ad(x::Call, new_tape)
    f = x.fn isa Variable ? new_tape[x.fn].val : x.fn
    f = f isa CoInstruction ? f : const_coinstruction(CoDual(f, uninit_tangent(f)))
    raw_args = map(x -> x isa Variable ? new_tape[x].val : x, x.args)
    args = map(raw_args) do x
        x isa CoInstruction ? x : const_coinstruction(CoDual(x, uninit_tangent(x)))
    end
    v = build_coinstruction(f, args...)
    return mkcall(v, f, args...; val=v)
end

struct UnrolledFunction
    tape::Tape
end


tangent_type(::Type{<:UnrolledFunction}) = NoTangent
randn_tangent(::AbstractRNG, ::UnrolledFunction) = NoTangent()
zero_tangent(::UnrolledFunction) = NoTangent()

(f::UnrolledFunction)(args...) = play!(f.tape, args...)

Base.show(io::IO, x::UnrolledFunction) = show(io, UnrolledFunction)

function seed_variable!(tape, var, ȳ)
    y_ref = tape[var].val.output
    dy = tangent(y_ref[])
    dy_new = increment!!(dy, ȳ)
    y_ref[] = CoDual(primal(y_ref[]), dy_new)
    return nothing
end

seed_instruction_output!(inst, ȳ) = seed_ref!(inst.output, ȳ)

function seed_ref!(y_ref, ȳ)
    dy = tangent(y_ref[])
    dy_new = increment!!(set_to_zero!!(dy), ȳ)
    y_ref[] = CoDual(primal(y_ref[]), dy_new)
    return nothing
end

function rebinding_pass!(tape)
    num_ops = length(tape)
    rebind_ops = Any[]
    for (i, op) in enumerate(reverse(tape.ops))
        op_num = num_ops - i + 1
        if op isa Umlaut.Call
            f_args = [op.fn, op.args...]
            new_args = map(enumerate(op.args)) do (n, arg)
                !(arg isa Variable) && return arg
                if findfirst(Base.Fix1(===, arg), f_args[1:n]) === nothing
                    return arg
                else
                    new_op = mkcall(rebind, arg; val=tape[arg].val)
                    push!(rebind_ops, new_op)
                    return Variable(new_op)
                end
            end
            if !isempty(rebind_ops)
                push!(rebind_ops, mkcall(op.fn, new_args...; val=op.val))
                replace!(tape, op_num => rebind_ops)
                empty!(rebind_ops)
            end
        end
    end
    return tape
end

"""
    literal_pass!(tape)

Replaces calls to `getfield(x, f)`, where `f` is a literal`, with
`lgetfield(x, SSym{f}())` or `lgetfield(x, SInt{f}())`, depending on the type of `f`.
This likely has little effect the first time that a function is differentiated, but should
make calls type-stable when re-run. This is useful when "compiling" the tape later on.
"""
function literal_pass!(tape)
    rebinding_dict = Dict()
    for (n, op) in enumerate(tape.ops)
        if op isa Call && op.fn == getfield && op.args[2] isa Symbol
            new_args = Any[op.args[1], SSym(op.args[2]), op.args[3:end]...]
            new_op = Call(op.id, op.val, lgetfield, new_args, op.tape, op.line)
            tape.ops[n] = new_op
            rebinding_dict[op.id] = new_op.id
        end
    end
    Umlaut.rebind!(tape, rebinding_dict)
    return tape
end

"""

"""
function intrinsic_pass!(tape)
    rebinding_dict = Dict()
    for (n, op) in enumerate(tape.ops)
        !(op isa Call) && continue
        f = op.fn isa Variable ? op.fn.op.val : op.fn
        if f isa Core.IntrinsicFunction
            new_fn = IntrinsicsWrappers.translate(Val(f))
            new_op = Call(op.id, op.val, new_fn, op.args, op.tape, op.line)
            tape.ops[n] = new_op
            rebinding_dict[op.id] = new_op.id
        end
    end
    Umlaut.rebind!(tape, rebinding_dict)
    return tape
end

function rrule_pass!(tape, args)
    inputs!(tape, map(const_coinstruction, args)...)
    new_tape = Tape(tape.c)
    for op in tape.ops
        new_op = to_reverse_mode_ad(op, new_tape)
        new_op_val = new_op.val.output[]
        if tangent_type(typeof(primal(new_op_val))) != typeof(tangent(new_op_val))
            inputs = map(getindex, new_op.val.inputs)
            display(inputs)
            println()
            display(new_op_val)
            println()
            display(which(rrule!!, map(Core.Typeof, inputs)))
            println()
            display("expected tangent type $(tangent_type(typeof(primal(new_op_val))))")
            println()
            throw(error("bad output types found in practice for op"))
        end
        push!(new_tape, new_op)
    end
    new_tape.result = Variable(new_tape[Variable(tape.result.id)])
    return new_tape
end

function rrule!!(f::CoDual{<:UnrolledFunction}, args...)
    tape = primal(f).tape
    tape = literal_pass!(tape)
    tape = intrinsic_pass!(tape)
    tape = rebinding_pass!(tape)

    rev_tape = rrule_pass!(tape, args)
    y_ref = rev_tape[rev_tape.result].val.output

    # Run the reverse-pass.
    function unrolled_function_pb!!(ȳ, ::NoTangent, dargs...)

        # Initialise values on the tape.
        seed_variable!(rev_tape, rev_tape.result, ȳ)
        foreach((v, x̄) -> seed_variable!(rev_tape, v, x̄), inputs(rev_tape), dargs)

        # Run the tape backwards.
        for op in reverse(rev_tape.ops)
            pullback!(rev_tape[Variable(op.id)].val)
        end

        # Extract the results from the tape.
        return NoTangent(), map(v -> tangent(rev_tape[v].val.output[]), inputs(rev_tape))...
    end

    return y_ref[], unrolled_function_pb!!
end

tangent_type(::Type{<:Umlaut.Variable}) = NoTangent

function value_and_gradient(tape::Tape, f, x...)
    f_ur = UnrolledFunction(tape)
    args = (f_ur, f, x...)
    dargs = map(zero_tangent, args)
    y, pb!! = rrule!!(map(CoDual, args, dargs)...)
    @assert primal(y) isa Float64
    dargs = pb!!(1.0, dargs...)
    return y, dargs
end

function value_and_gradient(f, x...)
    tape = last(trace(f, x...; ctx=RMC()))
    return value_and_gradient(tape, f, x...)
end

struct AcceleratedGradientTape{Ttape}
    tape::Ttape
end

function construct_accel_tape(f::CoDual, args::CoDual...)
    tape = primal(f).tape
    tape = literal_pass!(tape)
    tape = intrinsic_pass!(tape)
    tape = rebinding_pass!(tape)
    new_tape = rrule_pass!(tape, args)

    # Insert an additional input for the seed increment.
    y_ref = new_tape[new_tape.result].val.output
    output_cotangent_input = Input(1, zero_tangent(primal(y_ref[])), new_tape, 0)
    insert!(new_tape, 1, output_cotangent_input)

    # Push the seeding operation onto the tape after the forwards pass.
    seed_call = mkcall(
        seed_instruction_output!,
        new_tape.result,
        Variable(output_cotangent_input),
    )
    push!(new_tape, seed_call)

    # Push operations onto the tape to run the reverse-pass.
    for op in reverse(new_tape.ops[2:end-1])
        pb_op = mkcall(pullback!, Variable(op.id))
        push!(new_tape, pb_op)
        Umlaut.exec!(new_tape, pb_op)
    end

    # Accelerate the forwards-tape.
    return AcceleratedGradientTape(accelerate(new_tape))
end

function execute!(t::AcceleratedGradientTape, ȳ, x_x̄::CoDual...)

    # Set up the inputs.
    fast_tape = t.tape
    new_args = map(x_x̄, fast_tape.arg_refs[2:end]) do x_x̄, arg_ref
        arg_val = arg_ref[]
        arg_val.output[] = x_x̄
        return arg_val
    end

    # Run the tape.
    Taped.execute!(fast_tape, ȳ, new_args...)

    # Extract the results.
    d_args = map(fast_tape.arg_refs[2:end]) do arg_ref
        return tangent(arg_ref[].output[])
    end
    return NoTangent(), d_args...
end



#
# Recursive tape unrolling
#

struct AllPrimitiveContext <: TapedContext end
const APC = AllPrimitiveContext

# All things are primitives.
isprimitive(::APC, x...) = true
isprimitive(::APC, ::typeof(Umlaut.__new__), T, x...) = true
isprimitive(::APC, ::typeof(Umlaut.__foreigncall__), args...) = true
isprimitive(::APC, ::typeof(__intrinsic__), args...) = true
isprimitive(::APC, ::Core.Builtin, x...) = true

function trace_recursive_tape!!(f, args...)
    @nospecialize f args
    val, tape = Umlaut.trace(f, args...; ctx=APC())
    return val, UnrolledFunction(tape)
end

function Umlaut.trace_call!(t::Umlaut.Tracer{APC}, v_fargs...)
    @nospecialize t v_fargs
    return Umlaut.record_primitive!(t.tape, v_fargs...)
end

function Umlaut.record_primitive!(tape::Tape{APC}, v_fargs...)
    @nospecialize tape v_fargs
    line = get(tape.meta, :line, nothing)
    fargs = Any[v isa Variable ? v.op.val : v for v in v_fargs]
    if isprimitive(RMC(), fargs...)
        push!(tape, mkcall(v_fargs...; line=line))
    else
        val, uf = trace_recursive_tape!!(fargs...)
        push!(tape, mkcall(uf, v_fargs...; line, val))
    end
end
