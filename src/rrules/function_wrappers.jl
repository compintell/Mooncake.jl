# Type used to represent tangents of `FunctionWrapper`s. Also used to represent its fdata
# because `FunctionWrapper`s are mutable types.
mutable struct FunctionWrapperTangent{Tfwds_oc}
    fwds_wrapper::Tfwds_oc
    dobj_ref::Ref
end

function _construct_types(R, A)

    # Convert signature into a tuple of types.
    primal_arg_types = (A.parameters...,)

    # Signature and OpaqueClosure type for reverse pass.
    rvs_sig = Tuple{rdata_type(tangent_type(R))}
    primal_rdata_sig = Tuple{map(rdata_type ∘ tangent_type, primal_arg_types)...}
    pb_ret_type = Tuple{NoRData,primal_rdata_sig.parameters...}
    rvs_oc_type = Core.OpaqueClosure{rvs_sig,pb_ret_type}

    # Signature and OpaqueClosure type for forwards pass.
    fwd_sig = Tuple{map(fcodual_type, primal_arg_types)...}
    fwd_oc_type = Core.OpaqueClosure{fwd_sig,Tuple{fcodual_type(R),rvs_oc_type}}
    return fwd_oc_type, rvs_oc_type, fwd_sig, rvs_sig
end

function tangent_type(::Type{FunctionWrapper{R,A}}) where {R,A<:Tuple}
    return FunctionWrapperTangent{_construct_types(R, A)[1]}
end

import .TestUtils: has_equal_data_internal
function has_equal_data_internal(
    p::P, q::P, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {P<:FunctionWrapper}
    return has_equal_data_internal(p.obj, q.obj, equal_undefs, d)
end
function has_equal_data_internal(
    t::T, s::T, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T<:FunctionWrapperTangent}
    return has_equal_data_internal(t.dobj_ref[], s.dobj_ref[], equal_undefs, d)
end

function _function_wrapper_tangent(R, obj::Tobj, A, obj_tangent) where {Tobj}

    # Analyse types.
    _, _, fwd_sig, rvs_sig = _construct_types(R, A)

    # Construct reference to obj_tangent that we can read / write-to.
    obj_tangent_ref = Ref{tangent_type(Tobj)}(obj_tangent)

    # Contruct a rule for `obj`, applied to its declared argument types.
    rule = build_rrule(Tuple{Tobj,A.parameters...})

    # Construct stack which can hold pullbacks generated by `rule`. The forwards-pass will
    # run `rule` and push the pullback to `pb_stack`. The reverse-pass will pop and run it.
    pb_stack = Stack{pullback_type(typeof(rule), (Tobj, A.parameters...))}()

    # Construct reverse-pass. Note: this closes over `pb_stack`.
    run_rvs_pass = Base.Experimental.@opaque rvs_sig dy -> begin
        obj_rdata, dx... = pop!(pb_stack)(dy)
        obj_tangent_ref[] = increment_rdata!!(obj_tangent_ref[], obj_rdata)
        return NoRData(), dx...
    end

    # Construct fowards-pass. Note: this closes over the reverse-pass and `pb_stack`.
    run_fwds_pass = Base.Experimental.@opaque fwd_sig (x...) -> begin
        y, pb = rule(CoDual(obj, fdata(obj_tangent_ref[])), x...)
        push!(pb_stack, pb)
        return y, run_rvs_pass
    end

    t = FunctionWrapperTangent(run_fwds_pass, obj_tangent_ref)
    return t, obj_tangent_ref
end

function zero_tangent_internal(
    p::FunctionWrapper{R,A}, stackdict::Union{Nothing,IdDict}
) where {R,A}

    # If we've seen this primal before, then we must return that tangent.
    haskey(stackdict, p) && return stackdict[p]::tangent_type(typeof(p))

    # We have not seen this primal before, create it and log it.
    obj_tangent = zero_tangent_internal(p.obj[], stackdict)
    t, _ = _function_wrapper_tangent(R, p.obj[], A, obj_tangent)
    stackdict === nothing || setindex!(stackdict, t, p)
    return t
end

function randn_tangent_internal(
    rng::AbstractRNG, p::FunctionWrapper{R,A}, stackdict::Union{Nothing,IdDict}
) where {R,A}

    # If we've seen this primal before, then we must return that tangent.
    haskey(stackdict, p) && return stackdict[p]::tangent_type(typeof(p))

    # We have not seen this primal before, create it and log it.
    obj_tangent = randn_tangent_internal(rng, p.obj[], stackdict)
    t, _ = _function_wrapper_tangent(R, p.obj[], A, obj_tangent)
    stackdict === nothing || setindex!(stackdict, t, p)
    return t
end

function increment!!(t::T, s::T) where {T<:FunctionWrapperTangent}
    t.dobj_ref[] = increment!!(t.dobj_ref[], s.dobj_ref[])
    return t
end

function _set_to_zero!!(c::IncCache, t::FunctionWrapperTangent)
    t.dobj_ref[] = _set_to_zero!!(c, t.dobj_ref[])
    return t
end

function __add_to_primal(c::MaybeCache, p::FunctionWrapper, t::FunctionWrapperTangent, unsafe::Bool)
    return typeof(p)(__add_to_primal(c, p.obj[], t.dobj_ref[], unsafe))
end

function __diff(c::MaybeCache, p::P, q::P) where {R,A,P<:FunctionWrapper{R,A}}
    return first(_function_wrapper_tangent(R, p.obj[], A, __diff(c, p.obj[], q.obj[])))
end

function __dot(c::MaybeCache, t::T, s::T) where {T<:FunctionWrapperTangent}
    return __dot(c, t.dobj_ref[], s.dobj_ref[])
end

function __scale(c::MaybeCache, a::Float64, t::T) where {T<:FunctionWrapperTangent}
    return T(t.fwds_wrapper, Ref(__scale(c, a, t.dobj_ref[])))
end

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

# Will: to the best of my knowledge, no one has ever actually worked with FunctionWrappers
# before in the ChainRules ecosystem. Consequently, it shouldn't matter what type we use
# here. We might need to revise this is people start making use of FunctionWrappers in a
# meaningful way inside of ChainRules, but it seems unlikely that this will ever happen.
to_cr_tangent(t::FunctionWrapperTangent) = t

@is_primitive MinimalCtx Tuple{Type{<:FunctionWrapper},Any}
function rrule!!(::CoDual{Type{FunctionWrapper{R,A}}}, obj::CoDual{P}) where {R,A,P}
    t, obj_tangent_ref = _function_wrapper_tangent(R, obj.x, A, zero_tangent(obj.x, obj.dx))
    function_wrapper_pb(::NoRData) = NoRData(), rdata(obj_tangent_ref[])
    return CoDual(FunctionWrapper{R,A}(obj.x), t), function_wrapper_pb
end

@is_primitive MinimalCtx Tuple{<:FunctionWrapper,Vararg}
function rrule!!(f::CoDual{<:FunctionWrapper}, x::Vararg{CoDual})
    y, pb = f.dx.fwds_wrapper(x...)
    function_wrapper_eval_pb(dy) = pb(dy)
    return y, function_wrapper_eval_pb
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:function_wrappers})
    test_cases = Any[
        (false, :none, nothing, FunctionWrapper{Float64,Tuple{Float64}}, sin),
        (false, :none, nothing, FunctionWrapper{Float64,Tuple{Float64}}(sin), 5.0),
    ]
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:function_wrappers})
    test_cases = Any[
        (
            false,
            :none,
            nothing,
            function (x, y)
                p = FunctionWrapper{Float64,Tuple{Float64}}(x -> x * y)
                out = 0.0
                for _ in 1:1_000
                    out += p(x)
                end
                return out
            end,
            5.0,
            4.0,
        ),
        (
            false,
            :none,
            nothing,
            function (x::Vector{Float64}, y::Float64)
                p = FunctionWrapper{Float64,Tuple{Float64}}(x -> x * y)
                out = 0.0
                for _x in x
                    out += p(_x)
                end
                return out
            end,
            randn(100),
            randn(),
        ),
    ]
    return test_cases, Any[]
end
