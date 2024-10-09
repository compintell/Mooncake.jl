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
    equal_refs = Core.memoryrefoffset(x) == Core.memoryrefoffset(y)
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

_val(::Val{c}) where {c} = c

using Core: memoryref_isassigned, memoryrefget, memoryrefset!, memoryrefnew, memoryrefoffset

@zero_adjoint(
    MinimalCtx, Tuple{typeof(memoryref_isassigned), GenericMemoryRef, Symbol, Bool}
)

@inline Base.@propagate_inbounds function rrule!!(
    ::CoDual{typeof(memoryrefget)},
    x::CoDual{<:MemoryRef},
    _ordering::CoDual{Symbol},
    _boundscheck::CoDual{Bool},
)
    ordering = Val(primal(_ordering))
    bc = Val(primal(_boundscheck))
    dx = x.dx
    function memoryrefget_adjoint!!(dy)
        new_tangent = increment_rdata!!(memoryrefget(dx, _val(ordering), _val(bc)), dy)
        memoryrefset!(dx, new_tangent, _val(ordering), _val(bc))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    y = memoryrefget(x.x, _val(ordering), _val(bc))
    dy = fdata(memoryrefget(x.dx, _val(ordering), _val(bc)))
    return CoDual(y, dy), memoryrefget_adjoint!!
end

# Core.memoryrefmodify!

@inline function rrule!!(f::CoDual{typeof(memoryrefnew)}, x::CoDual{<:Memory})
    return CoDual(memoryrefnew(x.x), memoryrefnew(x.dx)), NoPullback(f, x)
end

@inline function rrule!!(
    f::CoDual{typeof(memoryrefnew)}, x::CoDual{<:MemoryRef}, ii::CoDual{Int}
)
    return CoDual(memoryrefnew(x.x, ii.x), memoryrefnew(x.dx, ii.x)), NoPullback(f, x, ii)
end

@inline function rrule!!(
    f::CoDual{typeof(memoryrefnew)},
    x::CoDual{<:MemoryRef},
    ii::CoDual{Int},
    boundscheck::CoDual{Bool},
)
    y = memoryrefnew(x.x, ii.x, boundscheck.x)
    dy = memoryrefnew(x.dx, ii.x, boundscheck.x)
    return CoDual(y, dy), NoPullback(f, x, ii, boundscheck)
end

@zero_adjoint MinimalCtx Tuple{typeof(memoryrefoffset), GenericMemoryRef}

# Core.memoryrefreplace!

@inline function rrule!!(
    ::CoDual{typeof(memoryrefset!)},
    x::CoDual{<:MemoryRef{P}, <:MemoryRef{V}},
    value::CoDual,
    _ordering::CoDual{Symbol},
    _boundscheck::CoDual{Bool},
) where {P, V}
    ordering = Val(_ordering.x)
    bc = Val(_boundscheck.x)

    isbitstype(P) && return isbits_memoryrefset!_rule(x, value, ordering, bc)

    to_save = isassigned(x.x)
    old_x = Ref{Tuple{P, V}}()
    if to_save
        old_x[] = (
            memoryrefget(x.x, _val(ordering), _val(bc)),
            memoryrefget(x.dx, _val(ordering), _val(bc)),
        )
    end

    memoryrefset!(x.x, value.x, _val(ordering), _val(bc))
    dx = x.dx
    memoryrefset!(dx, tangent(value.dx, zero_rdata(value.dx)), _val(ordering), _val(bc))
    function memoryrefset_adjoint!!(dy)
        dvalue = increment!!(dy, rdata(memoryrefget(dx, _val(ordering), _val(bc))))
        if to_save
            memoryrefset!(x.x, old_x[][1], _val(ordering), _val(bc))
            memoryrefset!(dx, old_x[][2], _val(ordering), _val(bc))
        end
        return NoRData(), NoRData(), dvalue, NoRData(), NoRData()
    end
    return value, memoryrefset_adjoint!!
end

function isbits_memoryrefset!_rule(x::CoDual, value::CoDual, ordering::Val, bc::Val)
    old_x = (
        memoryrefget(x.x, _val(ordering), _val(bc)),
        memoryrefget(x.dx, _val(ordering), _val(bc)),
    )
    memoryrefset!(x.x, value.x, _val(ordering), _val(bc))
    memoryrefset!(x.dx, zero_tangent(value.x), _val(ordering), _val(bc))

    function isbits_memoryrefset!_adjoint!!(dy)
        dvalue = increment!!(dy, rdata(memoryrefget(x.dx, _val(ordering), _val(bc))))
        memoryrefset!(x.x, old_x[1], _val(ordering), _val(bc))
        memoryrefset!(x.dx, old_x[2], _val(ordering), _val(bc))
        return NoRData(), NoRData(), dvalue, NoRData(), NoRData()
    end
    return value, isbits_memoryrefset!_adjoint!!
end

# Core.memoryrefsetonce!
# Core.memoryrefswap!
# Core.set_binding_type!


# Functionality for `Array`s. Since they behave like `mutable struct`s as of v1.11, they
# gain a range of additional functions (getfield, setfield!), etc. Since we use `Array`s
# as the tangent / fdata type for `Array`s, a range of functionality must be added to ensure
# that they interact correctly with these functions.

function rrule!!(
    f::CoDual{typeof(getfield)}, x::CoDual{<:Array}, name::CoDual{<:Union{Int, Symbol}}
)
    y = getfield(primal(x), primal(name))
    wants_size = primal(name) === 2 || primal(name) === :size
    dy = wants_size ? NoFData() : x.dx.ref
    return CoDual(y, dy), NoPullback(f, x, name)
end

@inline function rrule!!(
    ::CoDual{typeof(setfield!)}, value::CoDual{<:Array}, _name::CoDual, x::CoDual,
)
    name = _name.x
    old_x = getfield(value.x, name)
    old_dx = getfield(value.dx, name)
    setfield!(value.x, name, x.x)
    setfield!(value.dx, name, (name === :size || name === 2) ? x.x : x.dx)
    function array_setfield!_adjoint(::NoRData)
        setfield!(value.x, name, old_x)
        setfield!(value.dx, name, old_dx)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return x, array_setfield!_adjoint
end

# Test cases

function _mems()

    # Set up memory with an undefined element.
    mem_with_single_undef = Memory{Memory{Int}}(undef, 2)
    mem_with_single_undef[1] = fill!(Memory{Int}(undef, 4), 2)

    # Return a collection of test cases.
    mems = [
        fill!(Memory{Float64}(undef, 10), 0.0),
        fill!(Memory{Int}(undef, 5), 1),
        Memory{Vector{Float64}}([randn(1), randn(3)]),
        Memory{Vector{Float64}}(undef, 3),
        mem_with_single_undef,
    ]
    sample_values = [1.0, 3, randn(2), randn(2), Memory{Int}(undef, 5)]
    return mems, sample_values
end

function _mem_refs()
    mems, sample_values = _mems()
    mem_refs = vcat([memoryref(m) for m in mems], [memoryref(m, 2) for m in mems])
    return mem_refs, vcat(sample_values, sample_values)
end

function generate_data_test_cases(rng_ctor, ::Val{:memory})
    arrays = [
        randn(2),
    ]
    return vcat(_mems()[1], _mem_refs()[1], arrays)
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:memory})
    mems, _ = _mems()
    mem_refs, sample_mem_ref_values = _mem_refs()
    test_cases = vcat(
        [(false, :stability, nothing, memoryref_isassigned, mem_ref, :not_atomic, bc) for
            mem_ref in mem_refs for bc in [false, true]
        ],
        [(false, :stability, nothing, memoryrefget, mem_ref, :not_atomic, bc) for
            mem_ref in filter(isassigned, mem_refs) for bc in [false, true]
        ],
        [(false, :stability, nothing, memoryrefnew, mem) for mem in mems],
        [(false, :stability, nothing, memoryrefnew, mem, 1) for mem in mem_refs],
        [(false, :stability, nothing, memoryrefnew, mem, 1, bc) for
            mem in mem_refs for bc in [false, true]
        ],
        [(false, :stability, nothing, memoryrefoffset, mem_ref) for mem_ref in mem_refs],
        [
            (false, :stability, nothing, memoryrefset!, mem_ref, sample_value, :not_atomic, bc) for
            (mem_ref, sample_value) in zip(mem_refs, sample_mem_ref_values) for
            bc in [false, true]
        ],
        (false, :stability, nothing, setfield!, randn(10), :ref, randn(10).ref),
        (false, :stability, nothing, setfield!, randn(10), 1, randn(10).ref),
        (false, :stability, nothing, setfield!, randn(10), :size, (10, )),
        (false, :stability, nothing, setfield!, randn(10), 2, (10, )),
    )
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:memory})
    mems, _ = _mems()
    mem_refs, sample_mem_ref_values = _mem_refs()
    test_cases = Any[]
    memory = Any[]
    return test_cases, memory
end
