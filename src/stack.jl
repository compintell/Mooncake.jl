"""
    Stack{T}()

A stack specialised for reverse-mode AD.

Semantically equivalent to a usual stack, but never de-allocates memory once allocated.
"""
mutable struct Stack{T}
    memory::Vector{T}
    position::Int
    Stack{T}() where {T} = new{T}(Vector{T}(undef, 0), 0)
end

function Base.push!(x::Stack{T}, val::T) where {T}
    position = x.position + 1
    memory = x.memory
    x.position = position
    if position <= length(memory)
        @inbounds memory[position] = val
        return nothing
    else
        push!(memory, val)
        return nothing
    end
end

function Base.pop!(x::Stack)
    position = x.position
    val = x.memory[position]
    x.position = position - 1
    return val
end

Base.isempty(x::Stack) = x.position == 0

Base.length(x::Stack) = x.position
