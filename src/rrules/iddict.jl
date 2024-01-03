# We're going to use `IdDict`s to represent tangents for `IdDict`s.

tangent_type(::Type{<:IdDict{K, V}}) where {K, V} = IdDict{K, tangent_type(V)}
function randn_tangent(rng::AbstractRNG, d::IdDict{K, V}) where {K, V}
    return IdDict{K, tangent_type(V)}([k => randn_tangent(rng, v) for (k, v) in d])
end
function zero_tangent(d::IdDict{K, V}) where {K, V}
    return IdDict{K, tangent_type(V)}([k => zero_tangent(v) for (k, v) in d])
end

function increment!!(p::T, q::T) where {T<:IdDict}
    for k in keys(p)
        p[k] = increment!!(p[k], q[k])
    end
    return p
end
function set_to_zero!!(t::IdDict)
    foreach(keys(t)) do k
        t[k] = set_to_zero!!(t[k])
    end
    return t
end
function _scale(a::Float64, t::IdDict{K, V}) where {K, V}
    return IdDict{K, V}([k => _scale(a, v) for (k, v) in t])
end
_dot(p::T, q::T) where {T<:IdDict} = sum([_dot(p[k], q[k]) for k in keys(p)])
function _add_to_primal(p::IdDict{K, V}, t::IdDict{K}) where {K, V}
    ks = intersect(keys(p), keys(t))
    return IdDict{K, V}([k => _add_to_primal(p[k], t[k]) for k in ks])
end
function _diff(p::P, q::P) where {K, V, P<:IdDict{K, V}}
    @assert union(keys(p), keys(q)) == keys(p)
    return IdDict{K, tangent_type(V)}([k => _diff(p[k], q[k]) for k in keys(p)])
end
function TestUtils.populate_address_map!(m::TestUtils.AddressMap, p::IdDict, t::IdDict)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    foreach(n -> TestUtils.populate_address_map!(m, p[n], t[n]), keys(p))
    return m
end
function TestUtils.has_equal_data(p::P, q::P; equal_undefs=true) where {P<:IdDict}
    ks = union(keys(p), keys(q))
    ks != keys(p) && return false
    return all([TestUtils.has_equal_data(p[k], q[k]; equal_undefs) for k in ks])
end

# All of the rules in here are provided in order to avoid nasty `:ccall`s, and to support
# standard built-in functionality on `IdDict`s.

@is_primitive MinimalCtx Tuple{typeof(Base.rehash!), IdDict, Any}
function rrule!!(::CoDual{typeof(Base.rehash!)}, d::CoDual{<:IdDict}, newsz::CoDual)
    Base.rehash!(primal(d), primal(newsz))
    Base.rehash!(tangent(d), primal(newsz))
    return d, NoPullback()
end

@is_primitive MinimalCtx Tuple{typeof(setindex!), IdDict, Any, Any}
function rrule!!(::CoDual{typeof(setindex!)}, d::CoDual{IdDict{K,V}}, val, key) where {K, V}

    k = primal(key)
    restore_state = in(k, keys(primal(d)))
    if restore_state
        old_primal_val = primal(d)[k]
        old_tangent_val = tangent(d)[k]
    end

    setindex!(primal(d), primal(val), k)
    setindex!(tangent(d), tangent(val), k)

    function setindex_pb!!(_, df, dd, dval, dkey)

        # Increment tangent.
        dval = increment!!(dval, tangent(d)[k])

        # Restore previous state if necessary.
        if restore_state
            primal(d)[k] = old_primal_val
            tangent(d)[k] = old_tangent_val
        else
            delete!(primal(d), k)
            delete!(tangent(d), k)
        end

        return df, dd, dval, dkey
    end
    return d, setindex_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(get), IdDict, Any, Any}
function rrule!!(
    ::CoDual{typeof(get)}, d::CoDual{IdDict{K, V}}, key::CoDual, default::CoDual
) where {K, V}
    k = primal(key)
    has_key = in(k, keys(primal(d)))
    y = has_key ? CoDual(primal(d)[k], tangent(d)[k]) : default

    function get_pb!!(dy, df, dd, dkey, ddefault)
        if has_key
            dd[k] = increment!!(dd[k], dy)
        else
            ddefault = increment!!(ddefault, dy)
        end
        return df, dd, dkey, ddefault
    end
    return y, get_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(getindex), IdDict, Any}
function rrule!!(
    ::CoDual{typeof(getindex)}, d::CoDual{IdDict{K, V}}, key::CoDual
) where {K, V}
    k = primal(key)
    y = CoDual(getindex(primal(d), k), getindex(tangent(d), k))
    function getindex_pb!!(dy, df, dd, dkey)
        dd[k] = increment!!(dd[k], dy) 
        return df, dd, dkey
    end
    return y, getindex_pb!!
end

for name in [
    :(:jl_idtable_rehash), :(:jl_eqtable_put), :(:jl_eqtable_get), :(:jl_eqtable_nextind),
]
    @eval function rrule!!(::CoDual{typeof(_foreigncall_)}, ::CoDual{Val{$name}}, args...)
        unexepcted_foreigncall_error($name)
    end
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:iddict})
    test_cases = Any[
        (false, :stability, nothing, Base.rehash!, IdDict(true => 5.0, false => 4.0), 10),
        (false, :none, nothing, setindex!, IdDict(true => 5.0, false => 4.0), 3.0, false),
        (false, :none, nothing, setindex!, IdDict(true => 5.0), 3.0, false),
        (false, :none, nothing, get, IdDict(true => 5.0, false => 4.0), false, 2.0),
        (false, :none, nothing, get, IdDict(true => 5.0), false, 2.0),
        (false, :none, nothing, getindex, IdDict(true => 5.0, false => 4.0), true),
    ]
    memory = Any[]
    return test_cases, memory
end

generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:iddict}) = Any[], Any[]
