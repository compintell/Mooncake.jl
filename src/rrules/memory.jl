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
    isbitstype(T) && return sum(_map(dot, t, s))
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

# Rules for builtins which must be specially implemented for Memory because it uses a
# custom tangent type.

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

tangent_type(::Type{<:MemoryRef{P}}) where {P} = MemoryRef{tangent_type(P)}


# Functionality for `Array`s. Since they behave like `mutable struct`s as of v1.11, they
# gain a range of additional functions (getfield, setfield!), etc. Since we use `Array`s
# as the tangent / fdata type for `Array`s, a range of functionality must be added to ensure
# that they interact correctly with these functions.
