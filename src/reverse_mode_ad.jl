# This is a proof-of-concept prototype, not something for general use.
# It is not maintained, nor will PRs against it be accepted.

struct ReverseModeADContext <: TapedContext end

const RMC = ReverseModeADContext

struct CoDual{Tx, Tdx}
    x::Tx
    dx::Tdx
end

primal(x::CoDual) = x.x
shadow(x::CoDual) = x.dx

function verify_codual_type(::CoDual{P, T}) where {P, T}
    Tt = tangent_type(P)
    if Tt !== T
        throw(error("for primal of type $P, expected tangent of type $Tt, but found $T"))
    end
end

struct CoInstruction{Tinputs, Toutput, Tpb}
    inputs::Tinputs
    output::Toutput
    pb::Tpb
end

input_shadows(x::CoInstruction) = map(shadow ∘ getindex, x.inputs)
output_shadow(x::CoInstruction) = shadow(x.output[])

function build_coinstruction(inputs::CoInstruction...)
    input_refs = map(x -> x.output, inputs)
    input_values = map(getindex, input_refs)
    output_value, pb!! = rrule!!(input_values...)
    output_ref = Ref(output_value)
    pb_ref = Ref(pb!!)
    return CoInstruction(input_refs, output_ref, pb_ref)
end

function (instruction::CoInstruction)(inputs::CoInstruction...)
    input_refs = map(x -> x.output, inputs)
    input_values = map(getindex, input_refs)
    foreach(verify_codual_type, input_values)
    output_value, pb!! = rrule!!(input_values...)
    verify_codual_type(output_value)
    output_ref = instruction.output
    output_ref[] = output_value
    pb_ref = instruction.pb
    pb_ref[] = pb!!
    return CoInstruction(input_refs, output_ref, pb_ref)
end

function pullback!(instruction::CoInstruction)
    input_shadows = map(shadow ∘ getindex, instruction.inputs)
    output_shadow = shadow(instruction.output[])
    new_input_shadows = instruction.pb[](output_shadow, input_shadows...)
    foreach(update_shadow!, instruction.inputs, new_input_shadows)
    return nothing
end

pullback!(::CoInstruction{Nothing, <:Ref, Nothing}) = nothing

function update_shadow!(x::Ref{<:CoDual{Tx, Tdx}}, new_shadow::Tdx) where {Tx, Tdx}
    x_val = x[]
    x[] = CoDual(primal(x_val), new_shadow)
    return nothing
end

struct NoPullback end

@inline (::NoPullback)(dy, dx...) = dx

function rrule!!(::CoDual{typeof(verify)}, args...)
    return CoDual(verify(map(primal, args)...), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(Umlaut.__new__)}, xs...)
    y = Umlaut.__new__(map(primal, xs)...)
    P = primal(xs[1])
    dy = build_tangent(P, map(shadow, xs[2:end])...)
    function __new__pullback(dy, ::NoTangent, ::NoTangent, dxs...)
        new_dxs = map((x, y) -> increment!!(x, _value(y)), dxs, dy.fields)
        return NoTangent(), NoTangent(), new_dxs...
    end
    return CoDual(y, dy), __new__pullback
end

get_fields(x::Union{Tangent, MutableTangent}) = x.fields
get_fields(x) = x

#
# Core.Builtin -- these are "primitive" functions which must have rrules
#

function rrule!!(f::CoDual{<:Core.IntrinsicFunction}, x)
    if primal(f) === not_int
        return CoDual(not_int(primal(x)), NoTangent()), NoPullback()
    else
        throw(error("unknown unary Core.IntrinsicFunction $f"))
    end
end

function rrule!!(f::CoDual{<:Core.IntrinsicFunction}, args...)
    if primal(f) === sle_int
        return CoDual(sle_int(map(primal, args)...), NoTangent()), NoPullback()
    elseif primal(f) === slt_int
        return CoDual(slt_int(map(primal, args)...), NoTangent()), NoPullback()
    elseif primal(f) === sub_int
        return CoDual(sub_int(map(primal, args)...), NoTangent()), NoPullback()
    elseif primal(f) === add_int
        return CoDual(add_int(map(primal, args)...), NoTangent()), NoPullback()
    else
        throw(error("unknown Core.IntrinsicFunction $f"))
    end
end

# <:

function rrule!!(::CoDual{typeof(===)}, args...)
    return CoDual(===(map(primal, args)...), NoTangent()), NoPullback()
end

# Core._abstracttype
# Core._apply_iterate
# Core._apply_pure
# Core._call_in_world
# Core._call_in_world_total
# Core._call_latest
# Core._compute_sparams
# Core._equiv_typedef
# Core._expr
# Core._primitivetype
# Core._setsuper!
# Core._structtype
# Core._svec_ref
# Core._typebody!
# Core._typevar

function rrule!!(::CoDual{typeof(Core.apply_type)}, args...)
    arg_primals = map(primal, args)
    return CoDual(Core.apply_type(arg_primals...), NoTangent()), NoPullback()
end

# Core.arrayref
# Core.arrayset
# Core.arraysize
# Core.compilerbarrier
# Core.const_arrayref
# Core.donotdelete
# Core.finalizer
# Core.get_binding_type
# Core.ifelse
# Core.set_binding_type!
# Core.sizeof
# Core.svec
# applicable

function rrule!!(::CoDual{typeof(Core.fieldtype)}, args...)
    arg_primals = map(primal, args)
    return CoDual(Core.fieldtype(arg_primals...), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(getfield)}, value::CoDual, name::CoDual)
    _name = primal(name)
    function getfield_pullback(dy, ::NoTangent, dvalue, ::NoTangent)
        return NoTangent(), increment_field!!(dvalue, dy, _name), NoTangent()
    end
    y = CoDual(getfield(primal(value), _name), _getfield(shadow(value), _name))
    return y, getfield_pullback
end

# getglobal
# invoke
# isa
# isdefined
# modifyfield!
# nfields
# replacefield!

function _setfield!(value::MutableTangent, name, x)
    @set value.fields.$name = x
    return x
end

function rrule!!(::CoDual{typeof(setfield!)}, value, name, x)
    _name = primal(name)
    old_x = isdefined(primal(value), _name) ? getfield(primal(value), _name) : nothing
    function setfield!_pullback(dy, ::NoTangent, dvalue, ::NoTangent, dx)
        set_field_to_zero!!(dvalue, _name)
        new_dx = increment!!(dx, dy)
        old_x !== nothing && setfield!(primal(value), _name, old_x)
        return NoTangent(), dvalue, NoTangent(), new_dx
    end
    y = CoDual(
        setfield!(primal(value), _name, primal(x)),
        _setfield!(shadow(value), _name, shadow(x)),
    )
    return y, setfield!_pullback
end

# swapfield!
# throw

function rrule!!(::CoDual{typeof(tuple)}, args...)
    y = CoDual(tuple(map(primal, args)...), tuple(map(shadow, args)...))
    tuple_pullback(dy, ::NoTangent, dargs...) = NoTangent(), map(increment!!, dargs, dy)...
    return y, tuple_pullback
end

function rrule!!(::CoDual{typeof(typeassert)}, x, type)
    function typeassert_pullback(dy, ::NoTangent, dx, ::NoTangent)
        return NoTangent(), increment!!(dx, dy), NoTangent()
    end
    return CoDual(typeassert(primal(x), primal(type)), shadow(x)), typeassert_pullback
end

rrule!!(::CoDual{typeof(typeof)}, x) = CoDual(typeof(primal(x)), NoTangent()), NoPullback()

#
# (in-principle) non-essential rules
#

isprimitive(::RMC, ::typeof(sin), ::Float64) = true
function rrule!!(::CoDual{typeof(sin)}, x::CoDual{Float64})
    x = primal(x)
    partial = cos(x)
    function sin_pullback!!(dy::Float64, ::NoTangent, dx::Float64)
        return NoTangent(), increment!!(dx, dy * partial)
    end
    y = sin(x)
    return CoDual(y, zero(y)), sin_pullback!!
end

isprimitive(::RMC, ::typeof(cos), ::Float64) = true
function rrule!!(::CoDual{typeof(cos)}, x::CoDual{Float64})
    x = primal(x)
    partial = -sin(x)
    function cos_pullback!!(dy::Float64, ::NoTangent, dx::Float64)
        return NoTangent(), increment!!(dx, dy * partial)
    end
    y = cos(x)
    return CoDual(y, zero(y)), cos_pullback!!
end

isprimitive(::RMC, ::typeof(Base.getindex), x::Array, inds::Int...) = true
function rrule!!(
    ::CoDual{typeof(Base.getindex)},
    x::CoDual{<:Array, Tdx},
    inds::CoDual{Int}...,
) where {V, Tdx <: Array{V}}
    ind_primals = map(primal, inds)
    dx = shadow(x)
    y_primal = getindex(primal(x), ind_primals...)
    y_shadow = getindex(dx, ind_primals...)
    y = CoDual(y_primal, y_shadow)
    function getindex_pullback!!(dy::V, ::NoTangent, dx::Tdx, dinds::NoTangent...)
        setindex!(dx, increment!!(y_shadow, dy), ind_primals...)
        return NoTangent(), dx, dinds...
    end
    return y, getindex_pullback!!
end

isprimitive(::RMC, ::typeof(setindex!), x::Array, v, inds...) = true
function rrule!!(
    ::CoDual{typeof(Base.setindex!)},
    A::CoDual{<:Array, TdA},
    v::CoDual,
    inds::CoDual{Int}...,
) where {V, TdA <: Array{V}}
    ind_primals = map(primal, inds)
    old_A_v = getindex(primal(A), ind_primals...)
    old_A_v_t = getindex(shadow(A), ind_primals...)
    setindex!(primal(A), primal(v), ind_primals...)
    setindex!(shadow(A), shadow(v), ind_primals...)
    function setindex_pullback!!(dA::TdA, ::NoTangent, dA2::TdA, dv, dinds::NoTangent...)
        dv_new = increment!!(dv, getindex(dA, ind_primals...))
        setindex!(primal(A), old_A_v, ind_primals...)
        setindex!(dA, old_A_v_t, ind_primals...)
        return NoTangent(), dA, dv_new, dinds...
    end
    return A, setindex_pullback!!
end

#
# High-level
#

function to_reverse_mode_ad(tape::Tape{RMC}, inputs::CoInstruction...)
    inputs!(tape, inputs...)

    new_tape = Tape(tape.c)

    # Transform forwards pass, replacing ops with associated rrule calls.
    for op in tape.ops
        push!(new_tape, to_reverse_mode_ad(op, new_tape))
    end
    new_tape.result = unbind(tape.result)

    # Seed reverse-pass and create operations to execute it.
    seed_op = mkcall(seed_return!, new_tape.result)
    push!(new_tape, seed_op)
    Umlaut.exec!(new_tape, seed_op)
    for op in reverse(new_tape.ops[1:end-1])
        pb_op = mkcall(pullback!, Variable(op.id))
        push!(new_tape, pb_op)
        Umlaut.exec!(new_tape, pb_op)
    end
    return new_tape
end

const_coinstruction(x::CoDual) = CoInstruction(nothing, Ref(x), nothing)

to_reverse_mode_ad(x::Input, new_tape) = Input(x.val)
function to_reverse_mode_ad(x::Constant, new_tape)
    return Constant(const_coinstruction(CoDual(x.val, zero_tangent(x.val))))
end
function to_reverse_mode_ad(x::Call, new_tape)
    f = x.fn isa CoInstruction ? x.fn : const_coinstruction(CoDual(x.fn, zero_tangent(x.fn)))
    raw_args = map(x -> x isa Variable ? new_tape[x].val : x, x.args)
    args = map(raw_args) do x
        x isa CoInstruction ? x : const_coinstruction(CoDual(x, zero_tangent(x)))
    end
    return mkcall(build_coinstruction(f, args...), f, args...)
end

function seed_return!(x::CoInstruction{T, V}) where {T, V<:Ref{<:CoDual{Float64}}}
    output = x.output[]
    x.output[] = CoDual(primal(output), 1.0)
    return nothing
end
seed_return!(x) = throw(error("Expected CoInstruction scalar, got $(typeof(x))"))

function gradient(f, x)

    # Construct tape.
    _, tape = trace(f, copy(x); ctx=Taped.RMC())
    f_df = Taped.const_coinstruction(CoDual(f, NoTangent()))
    x_dx = Taped.const_coinstruction(CoDual(copy(x), zero(x)))
    rm_tape = to_reverse_mode_ad(tape, f_df, x_dx)

    # Construct gradient function.
    function _gradient(f, x)
        x_dx.output[] = CoDual(x, zero(x))
        play!(rm_tape, f_df, x_dx)
        return shadow(x_dx.output[])
    end

    return shadow(x_dx.output[]), _gradient
end

# # I need to implement this in order to have a consistent interface.
# function rrule!!(::CoDual{Tf}, x::CoDual...) where {Tf}

# end
