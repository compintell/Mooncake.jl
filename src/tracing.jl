abstract type TapedContext end

struct ConcreteTapedContext <: TapedContext end

const TC = ConcreteTapedContext

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

function Umlaut.handle_gotoifnot_node!(
    t::Tracer{<:TapedContext}, cf::Core.GotoIfNot, frame::Frame
)
    return if cf.cond isa Umlaut.Argument || cf.cond isa Umlaut.SSAValue
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
end

# By default, things are not primitive.
isprimitive(::TapedContext, ::F, x...) where {F} = false

# Umlaut versions of IR nodes and built-ins must be primitive.
isprimitive(::TapedContext, ::typeof(Umlaut.__new__), T, x...) = true
isprimitive(::TapedContext, ::Core.Builtin, x...) = true
