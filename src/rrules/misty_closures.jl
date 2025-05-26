struct MistyClosureTangent
    captures_tangent::Any
    dual_mc::MistyClosure
end

_dual_mc(p::MistyClosure) = build_frule(get_interpreter(), p)

tangent_type(::Type{<:MistyClosure}) = MistyClosureTangent

function zero_tangent_internal(p::MistyClosure, d::StackDict)
    return MistyClosureTangent(zero_tangent_internal(p.oc.captures, d), _dual_mc(p))
end

function randn_tangent_internal(rng::AbstractRNG, p::MistyClosure, d::StackDict)
    return MistyClosureTangent(randn_tangent_internal(rng, p.oc.captures, d), _dual_mc(p))
end

function increment_internal!!(c::IncCache, t::T, s::T) where {T<:MistyClosureTangent}
    new_captures_tangent = increment_internal!!(c, t.captures_tangent, s.captures_tangent)
    return MistyClosureTangent(new_captures_tangent, t.dual_mc)
end

function set_to_zero_internal!!(c::IncCache, t::MistyClosureTangent)
    new_captures_tangent = set_to_zero_internal!!(c, t.captures_tangent)
    return MistyClosureTangent(new_captures_tangent, t.dual_mc)
end

function _add_to_primal_internal(
    c::MaybeCache, p::MistyClosure, t::MistyClosureTangent, unsafe::Bool
)
    new_captures = _add_to_primal_internal(c, p.oc.captures, t.captures_tangent, unsafe)
    return replace_captures(p, new_captures)
end

function _diff_internal(c::MaybeCache, p::P, q::P) where {P<:MistyClosureTangent}
    # Just assumes that the code associated to `p` is the same as that of `q`.
    captures_tangent = _diff_internal(c, p.oc.captures, q.oc.captures)
    return MistyClosureTangent(captures_tangent, _dual_mc(p))
end

function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:MistyClosureTangent}
    return _dot_internal(c, t.oc.captures, s.oc.captures)
end

function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:MistyClosureTangent}
    captures_tangent = _scale_internal(c, a, t.oc.captures)
    return T(captures_tangent, t.dual_mc)
end

import .TestUtils: populate_address_map_internal, AddressMap
function populate_address_map_internal(
    m::AddressMap, p::MistyClosure, t::MistyClosureTangent
)
    return populate_address_map_internal(m, p.oc.captures, t.captures_tangent)
end

struct MistyClosureFData
    captures_fdata::Any
    dual_mc::MistyClosure
end

struct MistyClosureRData
    captures_rdata::Any
end

fdata_type(::Type{<:MistyClosureTangent}) = MistyClosureFData
rdata_type(::Type{<:MistyClosureTangent}) = MistyClosureRData
@foldable function tangent_type(::Type{<:MistyClosureFData}, ::Type{<:MistyClosureRData})
    return MistyClosureTangent
end
function tangent(f::MistyClosureFData, r::MistyClosureRData)
    return MistyClosureTangent(tangent(f.captures_data, r.captures_rdata), f.dual_mc)
end

function __verify_fdata_value(
    ::IdDict{Any,Nothing}, p::MistyClosure, t::MistyClosureTangent
)
    return nothing
end
