abstract type TapedContext end

struct ConcreteTapedContext <: TapedContext end

const TC = ConcreteTapedContext

struct ConditionalCheck{Tv}
    value::Tv
end

function verify(check::ConditionalCheck, new_value)
    check.value == new_value && return nothing
    throw(error(
        "Control flow has changed between the current execution and when this tape was " *
        "constructed. Please re-write your code to ensure that the same branches are " *
        "taken each time that your function is executed.",
    ))
end

function Umlaut.handle_gotoifnot_node!(
    t::Tracer{<:TapedContext}, cf::Core.GotoIfNot, frame::Frame
)
    @nospecialize t cf frame
    return if cf.cond isa Umlaut.Argument || cf.cond isa Umlaut.SSAValue
        # resolve tape var
        c = frame.ir2tape[cf.cond]
        !isa(c, Umlaut.Variable) && return c
        v = t.tape[c].val
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
end

__intrinsic__(::Val{f}, args::Vararg{Any, N}) where {f, N} = (f)(args...)

# By default, things are not primitive.
isprimitive(::TapedContext, ::F, x...) where {F} = false

# Umlaut versions of IR nodes and built-ins must be primitive.
isprimitive(::TapedContext, ::typeof(Umlaut.__new__), T, x...) = true
isprimitive(::TapedContext, ::typeof(Umlaut.__foreigncall__), args...) = true
isprimitive(::TapedContext, ::typeof(__intrinsic__), args...) = true
isprimitive(::TapedContext, ::Core.Builtin, x...) = true

unbind(v::Variable) = Variable(v.id)
unbind(v) = v


"""
    trace!(t::Tracer, v_fargs)

Trace call defined by variables in v_fargs.
"""
function Umlaut.trace!(t::Tracer{<:TapedContext}, v_fargs)
    @nospecialize t v_fargs
    v_fargs = Umlaut.unsplat!(t, v_fargs)
    # note: we need to extract IR before vararg grouping, which may change
    # v_fargs, thus invalidating method search
    sig = Umlaut.code_signature(t.tape.c, v_fargs)
    ir = Umlaut.getcode(sig...)
    sparams, sparams_dict = Umlaut.get_static_params(t, v_fargs)
    v_fargs = Umlaut.group_varargs!(t, v_fargs)
    frame = Umlaut.Frame(t.tape, ir, v_fargs...)
    push!(t.stack, frame)

    bi = 1
    prev_bi = 0
    cf = nothing
    while bi <= length(ir.cfg.blocks)
        cf = Umlaut.trace_block!(t, ir, bi, prev_bi, sparams, sparams_dict)
        if isnothing(cf)
            # fallthrough to the next block
            prev_bi = bi
            bi += 1
        elseif cf isa Core.GotoIfNot

            # conditional jump
            cond_val = Umlaut.handle_gotoifnot_node!(t, cf, frame)

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
            # line = "return value from $f$(map(Core.Typeof, args))"
            line = ""
            v = if res isa Umlaut.SSAValue || res isa Umlaut.Argument
                val = frame.ir2tape[res]
                val isa Umlaut.V ? val : push!(t.tape, Umlaut.Constant(Umlaut.promote_const_value(val); line))
            elseif Meta.isexpr(res, :static_parameter, 1)
                val = sparams[res.args[1]]
                push!(t.tape, Umlaut.Constant(Umlaut.promote_const_value(val); line))
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
    return Umlaut.V(t.tape[Umlaut.V(end)])
end
