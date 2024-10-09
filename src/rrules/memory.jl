#
# Memory
#

# Tangent Interface Implementation

const Maybe{T} = Union{Nothing, T}

tangent_type(::Type{<:Memory{P}}) where {P} = Memory{tangent_type(P)}

function zero_tangent_internal(x::Memory{P}, stackdict::Maybe{IdDict}) where {P}
    T = tangent_type(typeof(x))
    haskey(stackdict, x) && return stackdict[x]::T

    t = T(undef, length(x))
    stackdict[x] = t
    return _map_if_assigned!(Base.Fix2(zero_tangent_internal, stackdict), t, x)::T
end

function randn_tangent_internal(rng::AbstractRNG, x::Memory, stackdict::Maybe{IdDict})
    T = tangent_type(typeof(x))
    haskey(stackdict, x) && return stackdict[x]::T

    t = T(undef, length(x))
    stackdict[t] = t
    return _map_if_assigned!(x -> randn_tangent_internal(rng, x, stackdict), t, x)::T
end

function TestUtils.has_equal_data_internal(
    x::Memory{P}, y::Memory{P}, equal_undefs::Bool, d::Dict{Tuple{UInt, UInt}, Bool}
) where {P}
    length(x) == length(y) || return false
    id_pair = (objectid(x), objectid(y))
    haskey(d, id_pair) && return d[id_pair]

    d[id_pair] = true
    equality = map(1:length(x)) do n
        if isassigned(x, n) != isassigned(y, n)
            return !equal_undefs
        elseif !isassigned(x, n)
            return true
        else
            return TestUtils.has_equal_data_internal(x[n], y[n], equal_undefs, d)
        end
    end
    return all(equality)
end

function increment!!(x::Memory{P}, y::Memory{P}) where {P}
    return x === y ? x : _map_if_assigned!(increment!!, x, x, y)
end

set_to_zero!!(x::Memory) = _map_if_assigned!(set_to_zero!!, x, x)

function _add_to_primal(p::Memory{P}, t::Memory) where {P}
    return _map_if_assigned!(_add_to_primal, Memory{P}(undef, length(p)), p, t)
end

function _diff(p::Memory{P}, q::Memory{P}) where {P}
    return _map_if_assigned!(_diff, Memory{tangent_type(P)}(undef, length(p)), p ,q)
end

function _dot(t::Memory{T}, s::Memory{T}) where {T}
    isbitstype(T) && return sum(_map(_dot, t, s))
    return sum(
        _map(eachindex(t)) do n
            (isassigned(t, n) && isassigned(s, n)) ? _dot(t[n], s[n]) : 0.0
        end
    )
end

function _scale(a::Float64, t::Memory{T}) where {T}
    return _map_if_assigned!(Base.Fix1(_scale, a), Memory{T}(undef, length(t)), t)
end

import .TestUtils: populate_address_map!
function populate_address_map!(m::TestUtils.AddressMap, p::Memory, t::Memory)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    foreach(n -> isassigned(p, n) && populate_address_map!(m, p[n], t[n]), eachindex(p))
    return m
end

# FData / RData Interface Implementation

tangent_type(::Type{F}, ::Type{NoRData}) where {F<:Memory} = F

tangent(f::Memory, ::NoRData) = f

function _verify_fdata_value(p::Memory{P}, f::Memory{T}) where {P, T}
    @assert length(p) == length(f)
    return nothing
end

# Rules

function rrule!!(
    f::CoDual{typeof(getfield)}, x::CoDual{<:Memory}, name::CoDual{<:Union{Int, Symbol}}
)
    y = getfield(primal(x), primal(name))
    wants_length = primal(name) === 1 || primal(name) === :length
    dy = wants_length ? NoFData() : bitcast(Ptr{NoTangent}, x.dx.ptr)
    return CoDual(y, dy), NoPullback(f, x, name)
end



#
# MemoryRef
#

# Tangent Interface Implementation

tangent_type(::Type{<:MemoryRef{P}}) where {P} = MemoryRef{tangent_type(P)}

function zero_tangent_internal(x::MemoryRef, stackdict::Maybe{IdDict})
    t_mem = zero_tangent_internal(x.mem, stackdict)::Memory
    return memoryref(t_mem, Core.memoryrefoffset(x))
end

function randn_tangent_internal(rng::AbstractRNG, x::MemoryRef, stackdict::Maybe{IdDict})
    t_mem = randn_tangent_internal(rng, x.mem, stackdict)::Memory
    return memoryref(t_mem, Core.memoryrefoffset(x))
end

function TestUtils.has_equal_data_internal(
    x::MemoryRef{P}, y::MemoryRef{P}, equal_undefs::Bool, d::Dict{Tuple{UInt, UInt}, Bool}
) where {P}
    equal_refs = TestUtils.has_equal_data_internal(x[], y[], equal_undefs, d)
    equal_data = TestUtils.has_equal_data_internal(x.mem, y.mem, equal_undefs, d)
    return equal_refs && equal_data
end

function increment!!(x::MemoryRef{P}, y::MemoryRef{P}) where {P}
    return memoryref(increment!!(x.mem, y.mem), Core.memoryrefoffset(x))
end

function set_to_zero!!(x::MemoryRef)
    set_to_zero!!(x.mem)
    return x
end

function _add_to_primal(p::MemoryRef, t::MemoryRef)
    return memoryref(_add_to_primal(p.mem, t.mem), Core.memoryrefoffset(p))
end

function _diff(p::MemoryRef{P}, q::MemoryRef{P}) where {P}
    @assert Core.memoryrefoffset(p) == Core.memoryrefoffset(q)
    return memoryref(_diff(p.mem, q.mem), Core.memoryrefoffset(p))
end

function _dot(t::MemoryRef{T}, s::MemoryRef{T}) where {T}
    @assert Core.memoryrefoffset(t) == Core.memoryrefoffset(s)
    return _dot(t.mem, s.mem)
end

_scale(a::Float64, t::MemoryRef) = memoryref(_scale(a, t.mem), Core.memoryrefoffset(t))

function populate_address_map!(m::TestUtils.AddressMap, p::MemoryRef, t::MemoryRef)
    return populate_address_map!(m, p.mem, t.mem)
end

# FData / RData Interface Implementation

fdata_type(::Type{<:MemoryRef{T}}) where {T} = MemoryRef{T}

rdata_type(::Type{<:MemoryRef}) = NoRData

tangent_type(::Type{<:MemoryRef{T}}, ::Type{NoRData}) where {T} = MemoryRef{T}

tangent(f::MemoryRef, ::NoRData) = f

function _verify_fdata_value(p::MemoryRef{P}, f::MemoryRef{T}) where {P, T}
    @assert Core.memoryrefoffset(p) == Core.memoryrefoffset(f)
    _verify_fdata_value(p.mem, f.mem)
end

# Rules

function rrule!!(
    f::CoDual{typeof(getfield)}, x::CoDual{<:MemoryRef}, name::CoDual{<:Union{Int, Symbol}}
)
    y = getfield(primal(x), primal(name))
    wants_offset = primal(name) === 1 || primal(name) === :ptr_or_offset
    dy = wants_offset ? bitcast(Ptr{NoTangent}, x.dx.ptr_or_offset) : x.dx.mem
    return CoDual(y, dy), NoPullback(f, x, name)
end

#
# Rules for `Memory` and `MemoryRef`s
#

# @zero_adjoint(
#     MinimalCtx, Tuple{typeof(Core.memoryref_isassigned), GenericMemoryRef, Symbol, Bool}
# )

# Core.memoryref_isassigned
# Core.memoryrefget
# Core.memoryrefmodify!
# Core.memoryrefnew
# Core.memoryrefoffset
# Core.memoryrefreplace!
# Core.memoryrefset!
# Core.memoryrefsetonce!
# Core.memoryrefswap!
# Core.set_binding_type!


# Functionality for `Array`s. Since they behave like `mutable struct`s as of v1.11, they
# gain a range of additional functions (getfield, setfield!), etc. Since we use `Array`s
# as the tangent / fdata type for `Array`s, a range of functionality must be added to ensure
# that they interact correctly with these functions.
