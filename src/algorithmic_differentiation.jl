struct RMADContext
    pbs::Vector{Any}
end

RMADContext() = RMADContext([])

Umlaut.isprimitive(::RMADContext, ::F, x...) where {F} = false

taped_rrule(::F, x...) where {F} = nothing

# Rules for intrinsics.
# Intrinsics reference:
# https://github.com/JuliaLang/julia/blob/628209c1f2f746e3fc21ccd7cb34e67289403d44/src/intrinsics.cpp#L1207
# Possibly useful for relating to LLVM instructions.
Umlaut.isprimitive(::RMADContext, ::F, x...) where {F<:Core.IntrinsicFunction} = true

function taped_rrule(f::Core.IntrinsicFunction, x...)
    if f === Core.Intrinsics.sub_int
        sub_int_pullback(ȳ) = NoTangent(), NoTangent(), NoTangent()
        return f(x...), sub_int_pullback
    elseif f === Core.Intrinsics.slt_int
        slt_int_pullback(ȳ) = NoTangent(), NoTangent(), NoTangent()
        return f(x...), slt_int_pullback
    else
        throw(error("Unknown Core.Intrinsic function $f"))
    end
end

# Rules for built-ins.
Umlaut.isprimitive(::RMADContext, ::F, x...) where {F<:typeof(Core.apply_type)} = true

# Rules to work around limitations in Umlaut.

# Rules for differentiable functions.

for (M, f, arity) in DiffRules.diffrules(; filter_modules=[:Base])
    if arity == 1
        @eval Umlaut.isprimitive(::RMADContext, ::typeof($M.$(f)), ::Float64) = true
        @eval taped_rrule(::typeof($M.$(f)), x::Float64) = rrule($M.$(f), x)
    elseif arity == 2
        @eval Umlaut.isprimitive(::RMADContext, ::typeof($M.$(f)), ::Float64, ::Float64) = true
        @eval taped_rrule(::typeof($M.$(f)), x::Float64, y::Float64) = rrule($M.$(f), x, y)
    else
        throw(error("Expected arity = 1 or 2, got $arity"))
    end
end

struct ConditionalCheck{Tv}
    value::Tv
end

function verify(check::ConditionalCheck, new_value)
    check.value == new_value && return
    throw(error(
        "Control flow has changed between the current execution and when this tape was " *
        "constructed. Please re-write your code to ensure that the same branches are " *
        "taken each time that your function is executed.",
    ))
end

ChainRulesCore.@non_differentiable verify(::ConditionalCheck, ::Any)

"""
    trace!(t::Tracer, v_fargs)

Trace call defined by variables in v_fargs.
"""
function taped_trace!(t::Umlaut.Tracer, v_fargs)
    v_fargs = Umlaut.unsplat!(t, v_fargs)
    # note: we need to extract IR before vararg grouping, which may change
    # v_fargs, thus invalidating method search
    ir = Umlaut.getcode(Umlaut.code_signature(t.tape.c, v_fargs)...)
    v_fargs = Umlaut.group_varargs!(t, v_fargs)
    frame = Umlaut.Frame(t.tape, ir, v_fargs...)
    push!(t.stack, frame)
    sparams = Umlaut.get_static_params(t, v_fargs)
    bi = 1
    prev_bi = 0
    cf = nothing
    while bi <= length(ir.cfg.blocks)
        cf = Umlaut.trace_block!(t, ir, bi, prev_bi, sparams)
        if isnothing(cf)
            # fallthrough to the next block
            prev_bi = bi
            bi += 1
        elseif cf isa Core.GotoIfNot
            # conditional jump
            cond_val = if cf.cond isa Umlaut.Argument || cf.cond isa Umlaut.SSAValue
                # resolve tape var
                v = t.tape[frame.ir2tape[cf.cond]].val
                push!(t.tape, mkcall(verify, ConditionalCheck(v), frame.ir2tape[cf.cond]))
                v
            elseif cf.cond isa Bool
                # literal condition (e.g. while true)
                cf.cond
            elseif cf.cond == Expr(:boundscheck)
                # boundscheck expression must be evaluated on some later stage
                # of interpretation that we don't have access to on this level
                # so just skipping the check instead
                true
            else
                exc = AssertionError(
                    "Expected goto condition to be of type Argument, " *
                    "SSAValue or Bool, but got $(cf.cond). \n\nFull IR: \n\n$(ir)\n"
                )
                throw(exc)
            end
            # if not cond, set i to destination, otherwise step forward
            prev_bi = bi
            bi = !cond_val ? cf.dest : bi + 1
        elseif cf isa Core.GotoNode
            # unconditional jump
            prev_bi = bi
            bi = cf.label
        elseif cf isa Umlaut.ReturnNode
            # global STATE = t, cf, ir
            isdefined(cf, :val) || error("Reached unreachable")
            res = cf.val
            f, args... = Umlaut.var_values(v_fargs)
            line = "return value from $f$(map(Core.Typeof, args))"
            v = if res isa Umlaut.SSAValue || res isa Umlaut.Argument
                val = frame.ir2tape[res]
                val isa Umlaut.Variable ? val : push!(t.tape, Umlaut.Constant(Umlaut.promote_const_value(val); line))
            elseif Meta.isexpr(res, :static_parameter, 1)
                val = sparams[res.args[1]]
                push!(t.tape, Constant(Umlaut.promote_const_value(val); line))
            else
                push!(t.tape, Umlaut.Constant(Umlaut.promote_const_value(res); line))
            end
            pop!(t.stack)
            return v
        else
            error("Panic! Don't know how to handle control flow expression $cf")
        end
    end
    pop!(t.stack)
    # if no ReturnNode was encountered, use last op on the tape
    return Umlaut.Variable(t.tape[V(end)])
end

function taped_trace(f, args...; ctx)
    t = Umlaut.Tracer(Umlaut.Tape(ctx))
    v_fargs = Umlaut.inputs!(t.tape, f, args...)
    try
        rv = taped_trace!(t, v_fargs)
        t.tape.result = rv
        return t.tape[t.tape.result].val, t.tape
    catch
        Umlaut.LATEST_TRACER[] = t
        rethrow()
    end
end

function Umlaut.record_primitive!(tape::Tape{<:RMADContext}, v_fargs...)
    line = get(tape.meta, :line, nothing)
    v_f, v_args... = v_fargs
    f, args... = [v isa Umlaut.V ? tape[v].val : v for v in v_fargs]
    if taped_rrule(f, args...) === nothing
        throw(error("No rrule for primitive ($f, $args)"))
    else
        v_rrule = push!(tape, mkcall(taped_rrule, v_fargs...; line=line))
        v_val = push!(tape, mkcall(getindex, v_rrule, 1))
        v_pb = push!(tape, mkcall(getindex, v_rrule, 2))
        push!(tape.c.pbs, v_pb)
        return v_val
    end
end

function run_reverse_pass(ctx::RMADContext)
    
end

function value_and_derivative(f, x::Float64)
    ctx = RMADContext()
    val, tape = taped_trace(f, x; ctx=ctx)
    display(tape)
    println()
    display(ctx)
    println()
    return val, 0.0
end
