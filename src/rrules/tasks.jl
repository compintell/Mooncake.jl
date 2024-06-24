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

# function zero_tangent(p::Task)
#     return TaskTangent(
#         zero_tangent(p.next),
#         zero_tangent(p.queue),
#         zero_tangent(p.storage),
#         zero_tangent(p.donenotify),
#         zero_tangent(p.result),
#         zero_tangent(p.logstate),
#         zero_tangent(p.code),
#         ntuple(n -> NoTangent(), 9)...,
#     )
# end

# function randn_tangent(rng::AbstractRNG, p::Task)
#     return TaskTangent(
#         randn_tangent(rng, p.next),
#         randn_tangent(rng, p.queue),
#         randn_tangent(rng, p.storage),
#         randn_tangent(rng, p.donenotify),
#         randn_tangent(rng, p.result),
#         randn_tangent(rng, p.logstate),
#         randn_tangent(rng, p.code),
#         ntuple(n -> NoTangent(), 9)...,
#     )
# end

# function increment!!(t::TaskTangent, s::TaskTangent)
#     t === s && return t
#     t.next = increment!!(t.next, s.next)
#     t.queue = increment!!(t.queue, s.queue)
#     t.storage = increment!!(t.storage, s.storage)
#     t.donenotify = increment!!(t.donenotify, s.donenotify)
#     t.result = increment!!(t.result, s.result)
#     t.logstate = increment!!(t.logstate, s.logstate)
#     t.code = increment!!(t.code, s.code)
#     return t
# end

# function set_to_zero!!(t::TaskTangent)
#     t.next = set_to_zero!!(t.next)
#     t.queue = set_to_zero!!(t.queue)
#     t.storage = set_to_zero!!(t.storage)
#     t.donenotify = set_to_zero!!(t.donenotify)
#     t.result = set_to_zero!!(t.result)
#     t.logstate = set_to_zero!!(t.logstate)
#     t.code = set_to_zero!!(t.code)
#     return t
# end

# function _add_to_primal(p::Task, t::TaskTangent)
#     p.next = _add_to_primal(p.next, t.next)
#     p.queue = _add_to_primal(p.queue, t.queue)
#     p.storage = _add_to_primal(p.storage, t.storage)
#     p.donenotify = _add_to_primal(p.donenotify, t.donenotify)
#     p.result = _add_to_primal(p.result, t.result)
#     p.logstate = _add_to_primal(p.logstate, t.logstate)
#     p.code = _add_to_primal(p.code, t.code)
#     return p
# end


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

@is_primitive MinimalCtx Tuple{typeof(current_task)}
function rrule!!(f::CoDual{typeof(current_task)})
    return zero_fcodual(current_task()), NoPullback(f)
end

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
