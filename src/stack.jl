"""
    Stack{T}()

A stack specialised for reverse-mode AD.

Semantically equivalent to a usual stack, but never de-allocates memory once allocated.
"""
mutable struct Stack{T, Inc}
    const memory::Vector{T}
    position::Int
    Stack{T, Inc}() where {T, Inc} = new{T, Inc}(Vector{T}(undef, 0), 0)
end

Stack{T}() where {T} = Stack{T, 1}()

_copy(::Stack{T, Inc}) where {T, Inc} = Stack{T, Inc}()

@inline function Base.push!(x::Stack{T, Inc}, val::T) where {T, Inc}
    memory = x.memory
    position = x.position
    if rem(position, Inc) === 0

        # Since Inc is known at compile time, we can entirely avoid loading the length of
        # memory most the time, which is often quite helpful.
        l = length(memory)
        if position == l
            @noinline resize!(x.memory, l + Inc)
        end
    end
    new_position = position + 1
    x.position = new_position
    @inbounds memory[new_position] = val
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
