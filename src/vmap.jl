# This is a proof-of-concept prototype, not something for general use.
# It is not maintained, nor will PRs against it be accepted.

# Define batch types.
struct Batch{T, Tbatch}
    batch::Tbatch
    function Batch{T}(batch::Tbatch) where {T, Tbatch}
        return new{T, Tbatch}(batch)
    end
end

batch_type(::Float64) = Vector{Float64}

struct VMapContext end

const VMC = VMapContext

isprimitive(::VMC, f::F, x...) where {F} = isprimitive(TC(), f, x...)
function _v_eval(f::F, x...) where {F}
    try
        return f(x...)
    catch
        error("vectorisation not implemented for $f with argument types $((typeof.(x)..., ))")
    end
end


# Some basic primitives that we know how to vectorise.
isprimitive(::VMC, ::typeof(sin), x::Float64) = true
_v_eval(::typeof(sin), x::Batch{Float64}) = Batch{Float64}(map(sin, x.batch))

isprimitive(::VMC, ::typeof(cos), x::Float64) = true
_v_eval(::typeof(cos), x::Batch{Float64}) = Batch{Float64}(map(cos, x.batch))

function _v_eval(::typeof(Base.abs_float), x::Batch{Float64})
    return Batch{Float64}(map(Base.abs_float, x.batch))
end

# vectorise modifies a tape, based on the input arguments.
function vectorise(tape::Tape, args...)
    new_tape = Tape(tape.c)
    for (n, arg) in enumerate(args)
        a_type = typeof(tape.ops[n].val)
        @assert typeof(arg) == a_type || typeof(arg) <: Batch{a_type}
        push!(new_tape, Input(arg))
    end
    for op in tape.ops[length(args)+1:end]
        push!(new_tape, vectorise(op))
    end
    new_tape.result = unbind(tape.result)
    return new_tape
end

vectorise(x::Constant) = x
vectorise(x::Call) = mkcall(_v_eval, x.fn, map(unbind, x.args)...)

unbind(v::Variable) = Variable(v.id)
unbind(v) = v
