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
    val = x.memory[position]
    x.position = position - 1
    return val
end

struct SingletonStack{T} end

Base.push!(::SingletonStack, ::Any) = nothing
Base.pop!(::SingletonStack{T}) where {T} = T.instance


"""
    PtrStack{T}()

A stack in which point arithmetic is used to achieve better performance than a regular
`Stack`. This type should only be used when `T` is a bits type.
"""
mutable struct PtrStack{T}
    const memory::Vector{T}
    position::Ptr{T}
    finish::Ptr{T}
    function PtrStack{T}() where {T}
        @assert isbitstype(T)
        memory = Vector{T}(undef, 0)
        position = pointer(memory, 0)
        return new{T}(memory, position, position)
    end
end

_copy(::PtrStack{T}) where {T} = PtrStack{T}()

@inline function Base.push!(x::PtrStack{T}, val::T) where {T}
    position = x.position + Base.elsize(typeof(x.memory))
    if position <= x.finish # if we're not at the end of the vector yet
        unsafe_store!(position, val)
        x.position = position
        return nothing
    else
        @noinline return _update_ptr_stack(x, val)
    end
end

function _update_ptr_stack(x::PtrStack, val)
    memory = x.memory
    push!(memory, val)

    # Update the pointers to point to the correct place
    x.position = pointer(memory, length(memory))
    x.finish = pointer(memory, length(memory))
    return nothing
end

@inline function Base.pop!(x::PtrStack{T}) where {T}
    position = x.position
    val = unsafe_load(position)
    x.position = position - Base.elsize(typeof(x.memory))
    return val
end
