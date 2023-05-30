# This is a proof-of-concept prototype, not something for general use.
# It is not maintained, nor will PRs against it be accepted.

struct ReverseModeADContext <: TapedContext end

const RMC = ReverseModeADContext

struct Shadow{Tx, Tdx, Tpb}
    x::Tx
    dx::Ref{Tdx}
    pb!::Tpb
end

primal(x::Shadow) = x.x
shadow(x::Shadow) = x.dx

isprimitive(::RMC, ::typeof(sin), x::Float64) = true
function rrule(::typeof(sin), x::Shadow{Float64})
    dy = Ref(0.0)
    function sin_pullback!()
        shadow(x)[] += dy[] * cos(primal(x))
        return nothing
    end
    return Shadow(sin(primal(x)), dy, sin_pullback!)
end

isprimitive(::RMC, ::typeof(cos), x::Float64) = true
function rrule(::typeof(cos), x::Shadow{Float64})
    dy = Ref(0.0)
    function cos_pullback!()
        shadow(x)[] -= dy[] * sin(primal(x))
        return nothing
    end
    return Shadow(cos(primal(x)), dy, cos_pullback!)
end

# Non-participatory operations.
rrule(::typeof(>), a::Int, b::Int) = a > b
rrule(::typeof(-), a::Int, b::Int) = a - b
rrule(::Colon, a::Int, b::Int) = a:b
rrule(::typeof(iterate), x...) = iterate(x...)
rrule(::typeof(===), x, y) = x === y
function rrule(f::Core.IntrinsicFunction, x)
    f === Core.Intrinsics.not_int && return f(x)
    throw(error("unknown intrinsic $f"))
end
rrule(::typeof(getfield), x::Tuple, v::Int) = getfield(x, v)

function to_reverse_mode_ad(tape::Tape{RMC})
    new_tape = Tape(tape.c)

    # Transform forwards pass, replacing ops with associated rrule calls.
    for op in tape.ops
        push!(new_tape, to_reverse_mode_ad(op))
    end
    new_tape.result = unbind(tape.result)

    # Seed reverse-pass and create operations to execute it.
    push!(new_tape, mkcall(seed_return!, Variable(new_tape.result.id)))
    for op in reverse(new_tape.ops)
        push!(new_tape, mkcall(run_pullback!, Variable(op.id)))
    end
    return new_tape
end

to_reverse_mode_ad(x::Input) = x
to_reverse_mode_ad(x::Constant) = x
to_reverse_mode_ad(x::Call) = mkcall(rrule, x.fn, map(unbind, x.args)...)

function seed_return!(x::Shadow{Float64})
    shadow(x)[] = 1.0
end
seed_return!(x) = throw(error("Expected Shadow, got $(typeof(x))"))

function run_pullback!(x::Shadow)
    x.pb! === nothing && return nothing
    x.pb!()
    return nothing
end
run_pullback!(_) = nothing
