struct LogPdfContext <: TapedContext end

const LPC = LogPdfContext

assume(d::Distribution, name::Symbol) = rand(d)
isprimitive(::LPC, ::typeof(assume), x...) = true
isprimitive(::LPC, ::Type{<:Distribution}, x...) = true

function increment_logpdf(
    values::NamedTuple,
    current_logpdf::Ref{Float64},
    ::typeof(assume),
    d::Distribution,
    name::Symbol,
)
    current_logpdf[] += logpdf(d, values[name])
    return values[name]
end

increment_logpdf(::NamedTuple, ::Ref{Float64}, f, x...) = f(x...)

function to_logpdf(tape::Tape{LPC})
    new_tape = Tape(tape.c)
    push!(new_tape, Input(nothing))

    # Push the inputs onto the tape.
    nargs = length(inputs(tape))
    for n in 1:nargs
        push!(new_tape, tape.ops[n])
    end
    # logpdf_acc_op = push!(new_tape, Constant(Ref(0.0)))
    logpdf_acc_op = push!(new_tape, mkcall(Ref, 0.0))

    for op in tape.ops[nargs+1:end]
        push!(new_tape, to_logpdf(op, logpdf_acc_op))
    end
    new_tape.result = logpdf_acc_op
    return new_tape
end

to_logpdf(x::Constant, _) = x
function to_logpdf(x::Call, logpdf_acc_op)
    args = map(_increment, x.args)
    return mkcall(increment_logpdf, Variable(1), logpdf_acc_op, x.fn, args...)
end

_increment(x::Variable) = Variable(unbind(x).id + 2)
_increment(x) = x
