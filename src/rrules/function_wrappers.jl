mutable struct FunctionWrapperTangent{Tfwds_wrapper<:FunctionWrapper}
    fwds_wrapper::Tfwds_wrapper
end

function tangent_type(::Type{FunctionWrapper{R, A}}) where {R, A<:Tuple}
    primal_arg_types = (A.parameters..., )
    primal_rdata_sig = Tuple{map(rdata_type ∘ tangent_type, primal_arg_types)...}
    pb_ret_type = Tuple{NoRData, NoRData, primal_rdata_sig.parameters...}
    ret_rdata_type = rdata_type(tangent_type(R))
    pb_fw_type = FunctionWrapper{pb_ret_type, Tuple{ret_rdata_type}}
    fwds_ret_type = fwds_ret_type = Tuple{fcodual_type(R), pb_fw_type}
    primal_codual_sig = Tuple{map(fcodual_type, A.parameters)...}
    return FunctionWrapperTangent{FunctionWrapper{fwds_ret_type, primal_codual_sig}}
end

function zero_tangent_internal(p::FunctionWrapper, stackdict::Union{Nothing, IdDict})
    # If we've seen this primal before, then we must return that tangent.
    haskey(stackdict, p) && return stackdict[p]::T

    # We have not seen this primal before, create it and log it.
    t = FunctionWrapperTangent(p)
    stackdict[p] = t
    return t
end

function randn_tangent_internal(
    rng::AbstractRNG, p::FunctionWrapper, stackdict::Union{Nothing, IdDict}
)
    # If we've seen this primal before, then we must return that tangent.
    haskey(stackdict, p) && return stackdict[p]::T

    # We have not seen this primal before, create it and log it.
    t = FunctionWrapperTangent(p)
    stackdict[p] = t
    return t
end

increment!!(t::FunctionWrapperTangent, s::FunctionWrapperTangent) = t

set_to_zero!!(t::FunctionWrapperTangent) = t

_add_to_primal(p::FunctionWrapper, t::FunctionWrapperTangent, unsafe::Bool) = p

_diff(p::P, q::P) where {P<:FunctionWrapper} = FunctionWrapperTangent(p)

_dot(t::T, s::T) where {T<:FunctionWrapperTangent} = 0.0

_scale(a::Float64, t::FunctionWrapperTangent) = t

import .TestUtils: populate_address_map!, AddressMap
function populate_address_map!(m::AddressMap, p::FunctionWrapper, t::FunctionWrapperTangent)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    return m
end

fdata_type(T::Type{<:FunctionWrapperTangent}) = T
rdata_type(::Type{FunctionWrapperTangent}) = NoRData
tangent_type(F::Type{<:FunctionWrapperTangent}, ::Type{NoRData}) = F
tangent(f::FunctionWrapperTangent, ::NoRData) = f

_verify_fdata_value(p::FunctionWrapper, t::FunctionWrapperTangent) = nothing

_eval(obj_ref::Ref, args...) = obj_ref[](args...)

@is_primitive MinimalCtx Tuple{Type{<:FunctionWrapper}, Any}
function rrule!!(::CoDual{Type{FunctionWrapper{R, A}}}, obj::CoDual{P}) where {R, A, P}
    obj_ref = Ref(obj.x)

    # Create a reference to the primal-tangent pair associated to `obj`.
    F = fdata_type(tangent_type(Base.RefValue{P}))
    dobj = F == NoFData ? NoFData() : build_fdata(Base.RefValue{P}, (obj.x, ), (obj.dx, ))
    obj_ref_codual = CoDual(obj_ref, dobj)
    sig = Tuple{typeof(_eval), Base.RefValue{typeof(obj.x)}, A.parameters...}
    rule = build_rrule(sig)

    # Analyse types.
    primal_arg_types = (A.parameters..., )
    primal_codual_sig = Tuple{map(fcodual_type, A.parameters)...}
    primal_rdata_sig = Tuple{map(rdata_type ∘ tangent_type, primal_arg_types)...}
    ret_rdata_type = rdata_type(tangent_type(R))

    # Construct reverse-pass.
    pb_ref = Ref{pullback_type(typeof(rule), sig.parameters)}()
    run_rvs_pass(dy) = pb_ref[](dy)
    pb_ret_type = Tuple{NoRData, NoRData, primal_rdata_sig.parameters...}
    pb_fw_type = FunctionWrapper{pb_ret_type, Tuple{ret_rdata_type}}
    pb_fw = pb_fw_type(run_rvs_pass)

    # Construct fowards-pass.
    function run_fwds_pass(x::Vararg{CoDual})
        y, pb = rule(zero_fcodual(_eval), obj_ref_codual, x...)
        pb_ref[] = pb
        return y, pb_fw
    end

    fwds_ret_type = Tuple{fcodual_type(R), pb_fw_type}
    fwds_wrapper = FunctionWrapper{fwds_ret_type, primal_codual_sig}(run_fwds_pass)
    t = FunctionWrapperTangent(fwds_wrapper)
    function_wrapper_pb(::NoRData) = NoRData(), rdata(val(obj_ref_codual.dx.fields.x))
    return CoDual(FunctionWrapper{R, A}(obj.x), t), function_wrapper_pb
end

@is_primitive MinimalCtx Tuple{<:FunctionWrapper, Vararg}
function rrule!!(f::CoDual{<:FunctionWrapper}, x::Vararg{CoDual})
    y, pb = f.dx.fwds_wrapper(x...)
    function_wrapper_eval_pb(dy) = pb(dy)[2:end]
    return y, function_wrapper_eval_pb
end
