# This is a proof-of-concept prototype, not something for general use.
# It is not maintained, nor will PRs against it be accepted.

struct ReverseModeADContext end

const RMC = ReverseModeADContext

struct Shadow{Tx, Tdx, Tpb}
    x::Tx
    dx::Ref{Tdx}
    pb!::Tpb
end

primal(x::Shadow) = x.x
shadow(x::Shadow) = x.dx

isprimitive(::typeof(sin), x::Float64) = true
function rrule(::typeof(sin), x::Shadow{Float64})
    dy = Ref(0.0)
    function sin_pullback!()
        shadow(x)[] += dy[] * cos(x.x)
        return nothing
    end
    return Shadow(sin(x.x), dy, sin_pullback!)
end

isprimitive(::typeof(cos), x::Float64) = true
function rrule(::typeof(cos), x::Shadow{Float64})
    dy = Ref(0.0)
    function cos_pullback!()
        shadow(x)[] -= dy[] * sin(x.x)
        return nothing
    end
    return Shadow(cos(x.x), dy, cos_pullback!)
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

function to_reverse_mode_ad(tape::Tape{RMC}, args...)
    new_tape = Tape(tape.c)
    for (n, arg) in enumerate(args)
        a_type = typeof(tape.ops[n].val)
        @assert typeof(arg) == a_type || typeof(arg) <: Shadow{a_type}
        push!(new_tape, Input(arg))
    end
    for op in tape.ops[length(args)+1:end]
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

function seed_return!(x::Shadow{Float64})
    x.dx[] = 1.0
end
seed_return!(x) = throw(error("Expected Shadow, got $(typeof(x))"))

to_reverse_mode_ad(x::Constant) = x
to_reverse_mode_ad(x::Call) = mkcall(rrule, x.fn, map(unbind, x.args)...)

function run_pullback!(x::Shadow)
    x.pb! === nothing && return nothing
    x.pb!()
    return nothing
end
run_pullback!(_) = nothing
