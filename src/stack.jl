"""
    Stack{T}()

A stack specialised for reverse-mode AD.

Semantically equivalent to a usual stack, but never de-allocates memory once allocated.
"""
mutable struct Stack{T}
    const memory::Vector{T}
    position::Int
    Stack{T}() where {T} = new{T}(Vector{T}(undef, 0), 0)
end

_copy(::Stack{T}) where {T} = Stack{T}()

@inline function Base.push!(x::Stack{T}, val::T) where {T}
    position = x.position + 1
    memory = x.memory
    x.position = position
    if position <= length(memory)
        @inbounds memory[position] = val
    else
        @noinline push!(memory, val)
    end
    return nothing
end

@inline function Base.pop!(x::Stack)
    position = x.position
    @inbounds val = x.memory[position]
    x.position = position - 1
    return val
end

struct SingletonStack{T} end

Base.push!(::SingletonStack, ::Any) = nothing
@generated Base.pop!(::SingletonStack{T}) where {T} = T.instance


struct ImmutableStack{T}
    memory::Vector{T}
    length::Int
    position::Int
end

ImmutableStack{T}() where {T} = ImmutableStack{T}(Vector{T}(undef, 0), 0, 0)

_copy(::ImmutableStack{T}) where {T} = ImmutableStack{T}()

@inline function Base.push!(x::ImmutableStack{T}, val::T) where {T}
    position = x.position + 1
    memory = x.memory
    l = x.length
    if position <= l
        @inbounds memory[position] = val
    else
        @noinline push!(memory, val)
        l += 1
    end
    return ImmutableStack(memory, l, position)
end

@inline function Base.pop!(x::ImmutableStack)
    position = x.position
    memory = x.memory
    @inbounds val = memory[position]
    return val, ImmutableStack(memory, x.length, position - 1)
end
