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

struct NoPullback end

@inline (::NoPullback)(dy, dx...) = dx

function rrule!!(::CoDual{typeof(verify)}, args...)
    return CoDual(verify(map(primal, args)...), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(Umlaut.__new__)}, xs...)
    y = Umlaut.__new__(map(primal, xs)...)
    P = primal(xs[1])
    dy = build_tangent(P, map(shadow, xs[2:end])...)
    function __new__pullback(dy, d__new__, df, dxs...)
        new_dxs = map((x, y) -> increment!!(x, _value(y)), dxs, dy.fields)
        return d__new__, df, new_dxs...
    end
    function __new__pullback(dy::NamedTuple, d__new__, df, dxs...)
        new_dxs = map((x, y) -> increment!!(x, _value(y)), dxs, dy)
        return d__new__, df, new_dxs...
    end
    return CoDual(y, dy), __new__pullback
end

get_fields(x::Union{Tangent, MutableTangent}) = x.fields
get_fields(x) = x

#
# Core.Builtin -- these are "primitive" functions which must have rrules because no IR
# is available.
# There is a finite number of these functions.
# Any built-ins which don't have rules defined are left as comments with their names
# in this block of code. They will inevitably get implemented as we attempt to differentiate
# more and more code.
# The list of Core.IntrinsicFunction instances is not complete.
#

# This implementation for intrinsics is not at all performant.
# Unlike regular functions, intrinsics are _instances_ of `Core.IntrinsicFunction`,
# so they all have the same type, meaning that dispatch isn't possible.
# We'll need to produce re-write rules which replace any intrinsic function with a
# regular function that we can dispatch on. This is the same strategy as employed for
# IR nodes like `new`, `splatnew`, etc, so is clearly possible.
function rrule!!(f::CoDual{<:Core.IntrinsicFunction}, x)
    _f = primal(f)
    _x = primal(x)
    if _f === not_int
        return CoDual(not_int(_x), NoTangent()), NoPullback()
    else
        throw(error("unknown unary Core.IntrinsicFunction $f with argument $x"))
    end
end

# See comment above.
function rrule!!(f::CoDual{<:Core.IntrinsicFunction}, a::CoDual, b::CoDual)
    _f = primal(f)
    _a = primal(a)
    _b = primal(b)
    if _f === sitofp
        x = sitofp(_a, _b)
        return CoDual(x, zero_tangent(x)), NoPullback()
    elseif _f === sle_int
        return CoDual(sle_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === slt_int
        return CoDual(slt_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === sub_int
        return CoDual(sub_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === add_int
        return CoDual(add_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === add_float
        add_float_pb!!(c̄, ::NoTangent, ā, b̄) = NoTangent(), c̄ + ā, c̄ + b̄
        c = add_float(_a, _b)
        return CoDual(c, zero_tangent(c)), add_float_pb!!
    elseif _f === mul_float
        function mul_float_pb!!(c̄, ::NoTangent, ā, b̄)
            return NoTangent(), ā + c̄ * _b, b̄ + _a * c̄
        end
        c = mul_float(_a, _b)
        return CoDual(c, zero_tangent(c)), mul_float_pb!!
    elseif _f === eq_float
        return CoDual(eq_float(_a, _b), NoTangent()), NoPullback()
    elseif _f === bitcast
        v = bitcast(_a, _b)
        return CoDual(v, zero_tangent(v)), NoPullback()
    elseif _f === mul_int
        return CoDual(mul_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === and_int
        return CoDual(and_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === or_int
        return CoDual(or_int(_a, _b), NoTangent()), NoPullback()
    else
        throw(error("unknown Core.IntrinsicFunction $_f with args $((_a, _b))"))
    end
end

function rrule!!(::CoDual{typeof(<:)}, T1, T2)
    return CoDual(<:(primal(T1), primal(T2)), NoTangent()), NoPullback()
end

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

function rrule!!(::CoDual{typeof(Core.arraysize)}, X, dim)
    return CoDual(Core.arraysize(primal(X), primal(dim)), NoTangent()), NoPullback()
end

# Core.compilerbarrier
# Core.const_arrayref
# Core.donotdelete
# Core.finalizer
# Core.get_binding_type

function rrule!!(::CoDual{typeof(Core.ifelse)}, cond, a, b)
    _cond = primal(cond)
    function ifelse_pullback!!(dc, df, ::NoTangent, da, db)
        da = _cond ? increment!!(da, dc) : da
        db = _cond ? db : increment!!(db, dc)
        return df, NoTangent(), da, db
    end
    return ifelse(_cond, a, b), ifelse_pullback!!
end

# Core.set_binding_type!

function rrule!!(::CoDual{typeof(Core.sizeof)}, x)
    return CoDual(Core.sizeof(primal(x)), NoTangent()), NoPullback()
end

# Core.svec

function rrule!!(::CoDual{typeof(applicable)}, f, args...)
    return CoDual(applicable(primal(f), map(primal, args)...), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(Core.fieldtype)}, args...)
    arg_primals = map(primal, args)
    return CoDual(Core.fieldtype(arg_primals...), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(getfield)}, value::CoDual, name::CoDual)
    _name = primal(name)
    function getfield_pullback(dy, ::NoTangent, dvalue, ::NoTangent)
        new_dvalue = increment_field!!(dvalue, dy, _name)
        return NoTangent(), new_dvalue, NoTangent()
    end
    y = CoDual(getfield(primal(value), _name), _getfield(shadow(value), _name))
    return y, getfield_pullback
end

function rrule!!(::CoDual{typeof(getfield)}, value::CoDual, name::CoDual, order::CoDual)
    _name = primal(name)
    _order = primal(order)
    function getfield_pullback(dy, df, dvalue, dname, dorder)
        new_dvalue = increment_field!!(dvalue, dy, _name)
        return df, new_dvalue, dname, dorder
    end
    _order = _order isa Expr ? true : _order
    y = CoDual(
        getfield(primal(value), _name, _order),
        _getfield(shadow(value), _name, _order),
    )
    return y, getfield_pullback
end

# getglobal

function rrule!!(::CoDual{typeof(getglobal)}, a, b)
    v = getglobal(primal(a), primal(b))
    return CoDual(v, zero_tangent(v)), NoPullback()
end

# invoke

function rrule!!(::CoDual{typeof(isa)}, x, T)
    return CoDual(isa(primal(x), primal(T)), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(isdefined)}, args...)
    return CoDual(isdefined(map(primal, args)...), NoTangent()), NoPullback()
end

# modifyfield!

function rrule!!(::CoDual{typeof(nfields)}, x)
    return CoDual(nfields(primal(x)), NoTangent()), NoPullback()
end

# replacefield!

function _setfield!(value::MutableTangent, name, x)
    @set value.fields.$name = x
    return x
end

function rrule!!(::CoDual{typeof(setfield!)}, value, name, x)
    _name = primal(name)
    old_x = isdefined(primal(value), _name) ? getfield(primal(value), _name) : nothing
    function setfield!_pullback(dy, df, dvalue, ::NoTangent, dx)
        new_dx = increment!!(dx, getfield(dvalue.fields, _name).tangent)
        set_field_to_zero!!(dvalue, _name)
        new_dx = increment!!(new_dx, dy)
        old_x !== nothing && setfield!(primal(value), _name, old_x)
        return df, dvalue, NoTangent(), new_dx
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
# Rules to avoid foreigncall nodes
#

function rrule!!(
    ::CoDual{Type{T}}, ::CoDual{<:Type{<:UndefInitializer}}, args...
) where {T<:Array}
    A = T(undef, args...)
    return CoDual(A, zero_tangent(A)), NoPullback()
end



#
# Rules to avoid hitting limitations of Umlaut
# These should be removed at a later date.
#

isprimitive(::RMC, ::typeof(eltype), x) = true
function rrule!!(::CoDual{typeof(eltype)}, x)
    return CoDual(eltype(primal(x)), NoTangent()), NoPullback()
end

isprimitive(::RMC, ::typeof(Base.promote_op), x, S::Type...) = true
function rrule!!(::CoDual{typeof(Base.promote_op)}, args...)
    return CoDual(Base.promote_op(map(primal, args)...), NoTangent()), NoPullback()
end


#
# Rules to avoid / cope with Umlaut internals.
# We _might_ remove these at a later date if they turn out not to be needed.
#

function rrule!!(::CoDual{typeof(Umlaut.check_variable_length)}, args...)
    v = Umlaut.check_variable_length(map(primal, args)...)
    return CoDual(v, NoTangent()), NoPullback()
end

# Umlaut occassionally pushes `getindex` onto the tape.
# Easiest just to handle it like this.
# Might remove at a later date when `Umlaut.primitivize` works properly.
function rrule!!(::CoDual{typeof(getindex)}, x::CoDual{<:Tuple}, i::CoDual{Int})
    function getindex_pullback!!(dy, df, dx, ::NoTangent)
        dx = ntuple(n -> n == primal(i) ? increment!!(dx[n], dy) : dx[n], length(dx))
        return df, dx, NoTangent()
    end
    return CoDual(primal(x)[primal(i)], shadow(x)[primal(i)]), getindex_pullback!!
end



#
# (in-principle) non-essential rules.
# Some of these might be removed in the future in favour of lower-level rules.
#

isprimitive(::RMC, ::typeof(sin), ::Float64) = true
function rrule!!(::CoDual{typeof(sin)}, x::CoDual{Float64})
    x = primal(x)
    partial = cos(x)
    function sin_pullback!!(dy::Float64, dsin, dx::Float64)
        return dsin, increment!!(dx, dy * partial)
    end
    y = sin(x)
    return CoDual(y, zero(y)), sin_pullback!!
end

isprimitive(::RMC, ::typeof(cos), ::Float64) = true
function rrule!!(::CoDual{typeof(cos)}, x::CoDual{Float64})
    x = primal(x)
    partial = -sin(x)
    function cos_pullback!!(dy::Float64, dcos, dx::Float64)
        return dcos, increment!!(dx, dy * partial)
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
    y_shadow = zero_tangent(y_primal)
    y = CoDual(y_primal, y_shadow)
    function getindex_pullback!!(dy::V, df, dx::Tdx, dinds::NoTangent...)
        setindex!(dx, increment!!(getindex(dx, ind_primals...), dy), ind_primals...)
        return df, dx, dinds...
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
    function setindex_pullback!!(dA::TdA, df, dA2::TdA, dv, dinds::NoTangent...)
        dv_new = increment!!(dv, getindex(dA, ind_primals...))
        setindex!(primal(A), old_A_v, ind_primals...)
        setindex!(dA, old_A_v_t, ind_primals...)
        return df, dA, dv_new, dinds...
    end
    return A, setindex_pullback!!
end



#
# LinearAlgebra.BLAS
#

function _trans(flag, mat)
    flag === 'T' && return transpose(mat)
    flag === 'C' && return adjoint(mat)
    flag === 'N' && return mat
    throw(error("Unrecognised flag $flag"))
end

# This rule is potentially implemented at a level of abstraction that is a little too high.
# Currently views don't work properly. It might, therefore, make sense to directly implement
# a rule for the `ccall` that this function wraps.
# This requies improving `Umlaut` to make it handle foreigncall IR nodes properly.
isprimitive(::RMC, ::typeof(BLAS.gemm!), args...) = true
function rrule!!(
    ::CoDual{typeof(BLAS.gemm!)},
    tA::CoDual{<:AbstractChar},
    tB::CoDual{<:AbstractChar},
    alpha::CoDual{<:Union{Bool, T}},
    A::CoDual{<:AbstractVecOrMat{T}},
    B::CoDual{<:AbstractVecOrMat{T}},
    beta::CoDual{<:Union{Bool, T}},
    C::CoDual{<:AbstractVecOrMat{T}},
) where {T<:Union{Float32, Float64}}
    C_prev = copy(primal(C))
    shadow(C) .= 0
    _tA, _tB = primal(tA), primal(tB)
    α = primal(alpha)
    BLAS.gemm!(_tA, _tB, α, primal(A), primal(B), primal(beta), primal(C))
    function gemm!_pullback!!(dC_out, df, ::NoTangent, ::NoTangent, dalpha, dA, dB, dbeta, dC)

        # Restore previous state.
        primal(C) .= C_prev

        # Increment cotangents.
        dbeta += tr(dC' * primal(C))
        dalpha += tr(dC' * _trans(_tA, primal(A)) * _trans(_tB, primal(B)))
        dA .+= α * transpose(_trans(_tA, _trans(_tB, primal(B)) * transpose(dC)))
        dB .+= α * transpose(_trans(_tB, transpose(dC) * _trans(_tA, primal(A))))
        dC .*= primal(beta)

        return df, NoTangent(), NoTangent(), dalpha, dA, dB, dbeta, dC
    end
    return C, gemm!_pullback!!
end



#
# High-level -- AD functionality built on top of rules.
#

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

function to_reverse_mode_ad(tape::Tape{RMC}, ȳ, inputs::CoInstruction...)
    inputs!(tape, inputs...)

    new_tape = Tape(tape.c)

    # Transform forwards pass, replacing ops with associated rrule calls.
    for op in tape.ops
        push!(new_tape, to_reverse_mode_ad(op, new_tape))
    end
    new_tape.result = unbind(tape.result)

    # Seed reverse-pass and create operations to execute it.
    seed_op = mkcall(seed_return!, new_tape.result, ȳ)
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
    f = x.fn isa Variable ? new_tape[x.fn].val : x.fn
    f = f isa CoInstruction ? f : const_coinstruction(CoDual(f, zero_tangent(f)))
    raw_args = map(x -> x isa Variable ? new_tape[x].val : x, x.args)
    args = map(raw_args) do x
        x isa CoInstruction ? x : const_coinstruction(CoDual(x, zero_tangent(x)))
    end
    v = build_coinstruction(f, args...)
    return mkcall(v, f, args...; val=v)
end

function seed_return!(x::CoInstruction{T, V}, x̄) where {T, V}
    output = x.output[]
    x.output[] = CoDual(primal(output), x̄)
    return nothing
end

struct UnrolledFunction{Ttape}
    tape::Ttape
end


tangent_type(::Type{<:UnrolledFunction}) = NoTangent
randn_tangent(::AbstractRNG, ::UnrolledFunction) = NoTangent()
zero_tangnet(::UnrolledFunction) = NoTangent()

(f::UnrolledFunction)(args...) = play!(f.tape, args...)

function seed_variable!(tape, var, ȳ)
    y_ref = tape[var].val.output
    dy = shadow(y_ref[])
    dy_new = increment!!(dy, ȳ)
    y_ref[] = CoDual(primal(y_ref[]), dy_new)
    return nothing
end

function rrule!!(f::CoDual{<:UnrolledFunction}, args...)
    tape = primal(f).tape
    wrapped_args = map(const_coinstruction, args)
    inputs!(tape, wrapped_args...)

    new_tape = Tape(tape.c)

    # Transform forwards pass, replacing ops with associated rrule calls.
    for op in tape.ops
        push!(new_tape, to_reverse_mode_ad(op, new_tape))
    end
    new_tape.result = unbind(tape.result)
    y_ref = new_tape[new_tape.result].val.output

    # Run the reverse-pass.
    function unrolled_function_pb!!(ȳ, ::NoTangent, dargs...)

        # Initialise values on the tape.
        seed_variable!(new_tape, new_tape.result, ȳ)
        foreach((v, x̄) -> seed_variable!(new_tape, v, x̄), inputs(new_tape), dargs)

        # Run the tape backwards.
        for op in reverse(new_tape.ops)
            pullback!(new_tape[Variable(op.id)].val)
        end

        # Extract the results from the tape.
        return NoTangent(), map(v -> shadow(new_tape[v].val.output[]), inputs(new_tape))...
    end

    return y_ref[], unrolled_function_pb!!
end

tangent_type(::Type{<:Umlaut.Variable}) = NoTangent
