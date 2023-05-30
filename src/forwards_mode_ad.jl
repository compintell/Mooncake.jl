# This is a proof-of-concept prototype, not something for general use.
# It is not maintained, nor will PRs against it be accepted.

struct ForwardsModeADContext <: TapedContext end

const FMC = ForwardsModeADContext

struct Dual{Tx, Tdx}
    x::Tx
    dx::Tdx
end

isprimitive(::FMC, ::typeof(sin), x::Union{Float32, Float64}) = true
function frule(::typeof(sin), x::Dual{<:Union{Float32, Float64}})
    return Dual(sin(x.x), cos(x.x) * x.dx)
end

isprimitive(::FMC, ::typeof(cos), x::Union{Float32, Float64}) = true
function frule(::typeof(cos), x::Dual{<:Union{Float32, Float64}})
    return Dual(cos(x.x), -sin(x.x) * x.dx)
end

# Non-participatory operations.
frule(::typeof(>), a::Int, b::Int) = a > b
frule(::typeof(-), a::Int, b::Int) = a - b
frule(::Colon, a::Int, b::Int) = a:b
frule(::typeof(iterate), x...) = iterate(x...)
frule(::typeof(===), x, y) = x === y
function frule(f::Core.IntrinsicFunction, x...)
    f === Core.Intrinsics.not_int && return f(x...)
    throw(error("unknown intrinsic $f"))
end

function to_forwards_mode_ad(tape::Tape{FMC})
    new_tape = Tape(tape.c)
    for op in tape.ops
        push!(new_tape, to_forwards_mode_ad(op))
    end
    new_tape.result = unbind(tape.result)
    return new_tape
end

to_forwards_mode_ad(x::Input) = x
to_forwards_mode_ad(x::Constant) = x
to_forwards_mode_ad(x::Call) = mkcall(frule, x.fn, map(unbind, x.args)...)
