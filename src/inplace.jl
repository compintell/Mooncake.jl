struct InplaceContext <: TapedContext end

const IC = InplaceContext

struct InplaceData{T, Tdata}
    data::Tdata
    InplaceData{T}(data::Tdata) where {T, Tdata} = new{T, Tdata}(data)
end

isprimitive(::IC, ::typeof(*), A::StridedArray{Float64}, B::StridedArray{Float64}) = true

function InplaceData(::typeof(*), A::StridedArray{Float64}, B::StridedArray{Float64})
    return InplaceData{typeof(*)}((C=A*B, ))
end

function (data::InplaceData{typeof(*)})(A::StridedArray{Float64}, B::StridedArray{Float64})
    C = data.data.C
    LinearAlgebra.mul!(C, A, B)
    return C
end

function to_inplace(tape::Tape{<:IC})
    new_tape = Tape(tape.c)
    for op in tape.ops
        push!(new_tape, to_inplace(op, tape))
    end
    new_tape.result = unbind(tape.result)
    return new_tape
end

to_inplace(x::Input, _) = x
to_inplace(x::Constant, _) = x
function to_inplace(x::Call, tape)
    f = x.fn isa Variable ? tape[x.fn].val : x.fn
    args = map(x -> x isa Variable ? tape[x].val : x, x.args)
    return mkcall(InplaceData(f, args...), map(unbind, x.args)...)
end
