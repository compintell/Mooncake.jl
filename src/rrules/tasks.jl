# Tasks are recursively-defined, so their tangent type needs to be done manually.
# Occassionally one encountered tasks in code, but they don't actually get called. For
# example, calls to `rand` with a `TaskLocalRNG` will query the local task, purely for the
# sake of getting random number generator state associated to it.
# The goal of the code in this file is to ensure that this kind of usage of tasks is handled
# well, rather than attempting to properly handle tasks.

mutable struct TaskTangent end

tangent_type(::Type{Task}) = TaskTangent

zero_tangent(p::Task) = TaskTangent()

randn_tangent(rng::AbstractRNG, p::Task) = TaskTangent()

increment!!(t::TaskTangent, s::TaskTangent) = t

set_to_zero!!(t::TaskTangent) = t

_add_to_primal(p::Task, t::TaskTangent) = p

_diff(::Task, ::Task) = TaskTangent()

_dot(::TaskTangent, ::TaskTangent) = 0.0

_scale(::Float64, t::TaskTangent) = t

TestUtils.populate_address_map!(m::TestUtils.AddressMap, ::Task, ::TaskTangent) = m

fdata_type(::Type{TaskTangent}) = TaskTangent

rdata_type(::Type{TaskTangent}) = NoRData

tangent(t::TaskTangent, ::NoRData) = t

@inline function _get_fdata_field(_, t::TaskTangent, f...)
    f === (:rngState0, ) && return NoFData()
    f === (:rngState1, ) && return NoFData()
    f === (:rngState2, ) && return NoFData()
    f === (:rngState3, ) && return NoFData()
    f === (:rngState4, ) && return NoFData()
    throw(error("Unhandled field $f"))
end

@inline increment_field_rdata!(::TaskTangent, ::NoRData, ::Val) = nothing

function get_tangent_field(t::TaskTangent, f)
    f === :rngState0 && return NoTangent()
    f === :rngState1 && return NoTangent()
    f === :rngState2 && return NoTangent()
    f === :rngState3 && return NoTangent()
    f === :rngState4 && return NoTangent()
    throw(error("Unhandled field $f"))
end

set_tangent_field!(t::TaskTangent, f, ::NoTangent) = NoTangent()

@zero_adjoint MinimalCtx Tuple{typeof(current_task)}

_verify_fdata_value(::Task, ::TaskTangent) = nothing

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:tasks})
    test_cases = Any[
        (false, :none, nothing, lgetfield, Task(() -> nothing), Val(:rngState1)),
        (false, :none, nothing, getfield, Task(() -> nothing), :rngState1),
        (false, :none, nothing, lsetfield!, Task(() -> nothing), Val(:rngState1), UInt64(5)),
        (false, :stability_and_allocs, nothing, current_task),
    ]
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:tasks})
    test_cases = Any[
        (
            false, :none, nothing,
            (rng) -> (Random.seed!(rng, 0); rand(rng)), Random.default_rng(),
        ),
    ]
    memory = Any[]
    return test_cases, memory
end
