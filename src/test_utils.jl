module TestUtils

using JET, Random, Taped, Test, Umlaut
using Taped: CoDual, NoTangent, rrule!!, is_init, zero_codual

has_equal_data(x::T, y::T; equal_undefs=true) where {T<:String} = x == y
has_equal_data(x::Type, y::Type; equal_undefs=true) = x == y
has_equal_data(x::Core.TypeName, y::Core.TypeName; equal_undefs=true) = x == y
has_equal_data(x::Module, y::Module; equal_undefs=true) = x == y
function has_equal_data(x::T, y::T; equal_undefs=true) where {T<:Array}
    size(x) != size(y) && return false
    equality = map(1:length(x)) do n
        (isassigned(x, n) != isassigned(y, n)) && return !equal_undefs
        return (!isassigned(x, n) || has_equal_data(x[n], y[n]))
    end
    return all(equality)
end
has_equal_data(x::Float64, y::Float64; equal_undefs=true) = isapprox(x, y)
function has_equal_data(x::T, y::T; equal_undefs=true) where {T<:Core.SimpleVector}
    return all(map((a, b) -> has_equal_data(a, b; equal_undefs), x, y))
end
function has_equal_data(x::T, y::T; equal_undefs=true) where {T}
    isprimitivetype(T) && return isequal(x, y)
    return all(map(
        n -> isdefined(x, n) ? has_equal_data(getfield(x, n), getfield(y, n)) : true,
        fieldnames(T),
    ))
end
has_equal_data(x::T, y::T; equal_undefs=true) where {T<:Umlaut.Tape} = true

has_equal_data_up_to_undefs(x::T, y::T) where {T} = has_equal_data(x, y; equal_undefs=false)

const AddressMap = Dict{Ptr{Nothing}, Ptr{Nothing}}

"""
    populate_address_map(primal, tangent)

Constructs an empty `AddressMap` and calls `populate_address_map!`.
"""
populate_address_map(primal, tangent) = populate_address_map!(AddressMap(), primal, tangent)

"""
    populate_address_map!(m::AddressMap, primal, tangent)

Fills `m` with pairs mapping from memory addresses in `primal` to corresponding memory
addresses in `tangent`. If the same memory address appears multiple times in `primal`,
throws an `AssertionError` if the same address is not mapped to in `tangent` each time.
"""
function populate_address_map!(m::AddressMap, primal::P, tangent::T) where {P, T}
    isprimitivetype(P) && return m
    if ismutabletype(P)
        @assert T <: MutableTangent
        k = pointer_from_objref(primal)
        v = pointer_from_objref(tangent)
        haskey(m, k) && (@assert (m[k] == v))
        m[k] = v
    end
    foreach(fieldnames(P)) do n
        t_field = getfield(tangent.fields, n)
        if isdefined(primal, n) && is_init(t_field)
            populate_address_map!(m, getfield(primal, n), t_field.tangent)
        elseif isdefined(primal, n) && !is_init(t_field)
            throw(error("unhandled defined-ness"))
        elseif !isdefined(primal, n) && is_init(t_field)
            throw(error("unhandled defined-ness"))
        end
    end
    return m
end

function populate_address_map!(m::AddressMap, p::P, t) where {P<:Union{Tuple, NamedTuple}}
    foreach(n -> populate_address_map!(m, getfield(p, n), getfield(t, n)), fieldnames(P))
    return m
end

function populate_address_map!(m::AddressMap, p::Array, t::Array)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    foreach(n -> isassigned(p, n) && populate_address_map!(m, p[n], t[n]), eachindex(p))
    return m
end

function populate_address_map!(m::AddressMap, p::Core.SimpleVector, t::Vector{Any})
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    foreach(n -> populate_address_map!(m, p[n], t[n]), eachindex(p))
    return m
end

populate_address_map!(m::AddressMap, p::Union{Core.TypeName, Type, Symbol, String}, t) = m

"""
    address_maps_are_consistent(x::AddressMap, y::AddressMap)

`true` if all keys in both `x` and `y` map to the same values. i.e. if any key appears in
both `x` and `y`, and the corresponding value is not the same in both `x` and `y`, return
`false`.
"""
function address_maps_are_consistent(x::AddressMap, y::AddressMap)
    return all(map(k -> x[k] == y[k], collect(intersect(keys(x), keys(y)))))
end

# Assumes that the interface has been tested, and we can simply check for numerical issues.
function test_rrule_numerical_correctness(rng::AbstractRNG, f_f̄, x_x̄...)
    @nospecialize rng f_f̄ x_x̄

    x_x̄ = map(_deepcopy, x_x̄) # defensive copy

    # Run original function on deep-copies of inputs.
    f = primal(f_f̄)
    x = map(primal, x_x̄)
    x̄ = map(tangent, x_x̄)

    # Run primal, and ensure that we still have access to mutated inputs afterwards.
    x_primal = _deepcopy(x)
    y_primal = f(x_primal...)

    # Use finite differences to estimate vjps
    ẋ = randn_tangent(rng, x)
    ε = 1e-5
    x′ = _add_to_primal(x, _scale(ε, ẋ))
    y′ = f(x′...)
    ẏ = _scale(1 / ε, _diff(y′, y_primal))
    ẋ_post = _scale(1 / ε, _diff(x′, x_primal))

    # Run `rrule!!` on copies of `f` and `x`. We use randomly generated tangents so that we
    # can later verify that non-zero values do not get propagated by the rule.
    x_x̄_rule = map(x -> CoDual(_deepcopy(x), zero_tangent(x)), x)
    inputs_address_map = populate_address_map(map(primal, x_x̄_rule), map(tangent, x_x̄_rule))
    y_ȳ_rule, pb!! = rrule!!(f_f̄, x_x̄_rule...)

    # Verify that inputs / outputs are the same under `f` and its rrule.
    @test has_equal_data(x_primal, map(primal, x_x̄_rule))
    @test has_equal_data(y_primal, primal(y_ȳ_rule))

    # Query both `x_x̄` and `y`, because `x_x̄` may have been mutated by `f`.
    outputs_address_map = populate_address_map(
        (map(primal, x_x̄_rule)..., primal(y_ȳ_rule)),
        (map(tangent, x_x̄_rule)..., tangent(y_ȳ_rule)),
    )
    @test address_maps_are_consistent(inputs_address_map, outputs_address_map)

    # Run reverse-pass.
    ȳ_delta = randn_tangent(rng, primal(y_ȳ_rule))
    x̄_delta = map(Base.Fix1(randn_tangent, rng) ∘ primal, x_x̄_rule)

    ȳ_init = set_to_zero!!(tangent(y_ȳ_rule))
    x̄_init = map(set_to_zero!! ∘ tangent, x_x̄_rule)
    ȳ = increment!!(ȳ_init, ȳ_delta)
    x̄ = map(increment!!, x̄_init, x̄_delta)
    _, x̄... = pb!!(ȳ, tangent(f_f̄), x̄...)

    # Check that inputs have been returned to their original value.
    @test all(map(has_equal_data_up_to_undefs, x, map(primal, x_x̄_rule)))

    # pullbacks increment, so have to compare to the incremented quantity.
    @test _dot(ȳ_delta, ẏ) + _dot(x̄_delta, ẋ_post) ≈ _dot(x̄, ẋ) rtol=1e-3 atol=1e-3
end

get_address(x) = ismutable(x) ? pointer_from_objref(x) : nothing

_deepcopy(x) = deepcopy(x)
_deepcopy(x::Module) = x

rrule_output_type(::Type{Ty}) where {Ty} = Tuple{CoDual{Ty, tangent_type(Ty)}, Any}

# Central definition of _typeof in case I need to specalise it for particular types.
_typeof(x::T) where {T} = Core.Typeof(x)

function test_rrule_interface(f_f̄, x_x̄...; is_primitive)
    @nospecialize f_f̄ x_x̄

    # Pull out primals and run primal computation.
    f = primal(f_f̄)
    f̄ = tangent(f_f̄)
    x_x̄ = map(_deepcopy, x_x̄)
    x = map(primal, x_x̄)
    x̄ = map(tangent, x_x̄)

    # Verify that the function to which the rrule applies is considered a primitive.
    # It is not clear that this really belongs here to be frank.
    is_primitive && @test Umlaut.isprimitive(Taped.RMC(), f, deepcopy(x)...)

    # Run the primal programme. Bail out early if this doesn't work.
    y = try
        f(deepcopy(x)...)
    catch e
        display(e)
        println()
        throw(ArgumentError("Primal evaluation does not work."))
    end

    # Check that input types are valid.
    @test _typeof(tangent(f_f̄)) == tangent_type(_typeof(primal(f_f̄)))
    for x_x̄ in x_x̄
        @test _typeof(tangent(x_x̄)) == tangent_type(_typeof(primal(x_x̄)))
    end

    # Run the rrule, check it has output a thing of the correct type, and extract results.
    # Throw a meaningful exception if the rrule doesn't run at all.
    x_addresses = map(get_address, x)
    rrule_ret = try
        rrule!!(f_f̄, x_x̄...)
    catch e
        display(e)
        println()
        throw(ArgumentError(
            "rrule!! for $(_typeof(f_f̄)) with argument types $(_typeof(x_x̄)) does not run."
        ))
    end
    @test rrule_ret isa rrule_output_type(_typeof(y))
    y_ȳ, pb!! = rrule_ret

    # Run the reverse-pass. Throw a meaningful exception if it doesn't run at all.
    f̄_new, x̄_new... = try
        pb!!(tangent(y_ȳ), f̄, x̄...)
    catch e
        display(e)
        println()
        throw(ArgumentError(
            "pullback for $(_typeof(f_f̄)) with argument types $(_typeof(x_x̄)) does not run."
        ))
    end

    # Check that memory addresses have remained constant under pb!!.
    new_x_addresses = map(get_address, x)
    @test all(map(==, x_addresses, new_x_addresses))

    # Check the tangent types output by the reverse-pass, and that memory addresses of
    # mutable objects have remained constant.
    @test _typeof(f̄_new) == _typeof(f̄)
    @test all(map((a, b) -> _typeof(a) == _typeof(b), x̄_new, x̄))
    @test all(map((x̄, x̄_new) -> ismutable(x̄) ? x̄ === x̄_new : true, x̄, x̄_new))
end

function test_rrule_performance(performance_checks_flag::Symbol, f_f̄, x_x̄...)
    @nospecialize f_f̄ x_x̄

    # Verify that a valid performance flag has been passed.
    valid_flags = (:none, :stability)
    if !in(performance_checks_flag, valid_flags)
        throw(ArgumentError(
            "performance_checks=$performance_checks_flag. Must be one of $valid_flags"
        ))
    end
    performance_checks_flag == :none && return nothing

    if performance_checks_flag == :stability

        # Test primal stability.
        JET.test_opt(primal(f_f̄), map(_typeof ∘ primal, x_x̄))

        # Test forwards-pass stability.
        JET.test_opt(rrule!!, (typeof(f_f̄), map(_typeof, x_x̄)...))

        # Test reverse-pass stability.
        y_ȳ, pb!! = rrule!!(f_f̄, _deepcopy(x_x̄)...)
        JET.test_opt(
            pb!!,
            (_typeof(tangent(y_ȳ)), _typeof(tangent(f_f̄)), map(_typeof ∘ tangent, x_x̄)...),
        )
    end
end

"""
    test_rrule!!(
        rng::AbstractRNG, x...;
        interface_only=false, is_primitive=true, perf_flag::Symbol,
    )

Run standardised tests on the `rrule!!` for `x`.
The first element of `x` should be the primal function to test, and each other element a
positional argument.
In most cases, elements of `x` can just be the primal values, and `randn_tangent` can be
relied upon to generate an appropriate tangent to test. Some notable exceptions exist
though, in partcular `Ptr`s. In this case, the argument for which `randn_tangent` cannot be
readily defined should be a `CoDual` containing the primal, and a _manually_ constructed
tangent field.
"""
function test_rrule!!(rng, x...; interface_only=false, is_primitive=true, perf_flag::Symbol)
    @nospecialize rng x

    # Generate random tangents for anything that is not already a CoDual.
    # x_x̄ = map(x -> x isa CoDual ? x : CoDual(x, randn_tangent(rng, x)), x)
    x_x̄ = map(x -> x isa CoDual ? x : zero_codual(x), x)

    # Test that the interface is basically satisfied (checks types / memory addresses).
    test_rrule_interface(x_x̄...; is_primitive)

    # Test that answers are numerically correct / consistent.
    interface_only || test_rrule_numerical_correctness(rng, x_x̄...)

    # Test the performance of the rule.
    test_rrule_performance(perf_flag, x_x̄...)
end

# Functionality for testing AD via Umlaut.
function test_taped_rrule!!(rng, f, x...; interface_only=false, recursive=true, kwargs...)
    @nospecialize rng f x

    # Try to run the primal, just to make sure that we're not calling it on bad inputs.
    f(_deepcopy(x)...)

    # Construct the tape.
    if recursive
        f_t = last(Taped.trace_recursive_tape!!(f, map(_deepcopy, x)...))
    else
        f_t = Taped.UnrolledFunction(last(trace(f, map(_deepcopy, x)...; ctx=Taped.RMC())))
    end

    # Check that the gradient is self-consistent.
    test_rrule!!(
        rng, f_t, f, x...; is_primitive=false, perf_flag=:none, interface_only, kwargs...
    )

    # Check that f_t remains a faithful representation of the original function.
    if !interface_only

        xs_lhs = deepcopy(x)
        xs_rhs = deepcopy(x)
        y_lhs = f(xs_lhs...)
        y_rhs = play!(f_t.tape, f, xs_rhs...)
        if !has_equal_data(y_lhs, y_rhs) || !has_equal_data(xs_lhs, xs_rhs)

            println("f")
            display(f)
            println()

            println("y_lhs")
            display(y_lhs)
            println()
            println("y_rhs")
            display(y_rhs)
            println()

            println("xs_lhs")
            display(xs_lhs)
            println()
            println("xs_rhs")
            display(xs_rhs)
            println()
        end
        @test has_equal_data(y_lhs, y_rhs)
        @test has_equal_data(xs_lhs, xs_rhs)
    end
end



#
# Test that some basic operations work on a given type.
#

generate_args(::typeof(===), x) = [(x, 0.0), (1.0, x)]
function generate_args(::typeof(Core.ifelse), x)
    return [(true, x, 0.0), (false, x, 0.0), (true, 0.0, x), (false, 0.0, x)]
end
generate_args(::typeof(Core.sizeof), x) = [(x, )]
generate_args(::typeof(Core.svec), x) = [(x, ), (x, x)]
function generate_args(::typeof(getfield), x)
    names = filter(f -> isdefined(x, f), fieldnames(typeof(x)))
    return map(n -> (x, n), vcat(names..., eachindex(names)...))
end
generate_args(::typeof(isa), x) = [(x, Float64), (x, Int), (x, typeof(x))]
function generate_args(::typeof(setfield!), x)
    names = filter(f -> isdefined(x, f), fieldnames(typeof(x)))
    return map(n -> (x, n, getfield(x, n)), vcat(names..., eachindex(names)...))
end
generate_args(::typeof(tuple), x) = [(x, ), (x, x), (x, x, x)]
generate_args(::typeof(typeassert), x) = [(x, typeof(x))]
generate_args(::typeof(typeof), x) = [(x, )]

function functions_for_all_types()
    return [===, Core.ifelse, Core.sizeof, isa, tuple, typeassert, typeof]
end

functions_for_structs() = vcat(functions_for_all_types(), [getfield])

function functions_for_mutable_structs()
    return vcat(
        functions_for_structs(), [setfield!],# modifyfield!, replacefield!, swapfield!]
    )
end

"""
    test_rule_and_type_interactions(rng::AbstractRNG, x)

Check that a collection of standard functions for which we _ought_ to have a working rrule
for `x` work, and produce the correct answer. For example, the `rrule!!` for `typeof` should
work correctly on any type, we should have a working rule for `getfield` for any
struct-type, and we should have a rule for `setfield!` for any mutable struct type.

The purpose of this test is to ensure that, for any given `x`, the full range of primitive
functions that _ought_ to work on it, do indeed work on it.
"""
function test_rule_and_type_interactions(rng::AbstractRNG, x::P) where {P}
    @nospecialize rng x

    # Generate standard test cases.
    fs = if ismutabletype(P)
        functions_for_mutable_structs()
    elseif isstructtype(P)
        functions_for_structs()
    else
        functions_for_all_types()
    end

    # Run standardised tests for all functions.
    @testset "$f" for f in fs
        arg_sets = generate_args(f, x)
        @testset for args in arg_sets
            test_rrule!!(
                rng, f, args...;
                interface_only=false,
                is_primitive=true,
                perf_flag=:none,
            )
        end
    end
end

"""
    test_tangent(rng::AbstractRNG, p::P, z_target::T, x::T, y::T) where {P, T}

Verify that primal `p` with tangents `z_target`, `x`, and `y`, satisfies the tangent
interface. If these tests pass, then it should be possible to write `rrule!!`s for primals
of type `P`, and to test them using `test_rrule!!`.

As always, there are limits to the errors that these tests can identify -- they form
necessary but not sufficient conditions for the correctness of your code.
"""
function test_tangent(rng::AbstractRNG, p::P, z_target::T, x::T, y::T) where {P, T}
    @nospecialize rng p z_target x y

    # This basic functionality must run in order to be able to check everything else.
    @test tangent_type(P) isa Type
    @test tangent_type(P) == T
    @test zero_tangent(p) isa T
    @test randn_tangent(rng, p) isa T
    test_equality_comparison(p)
    test_equality_comparison(x)

    # Verify that interface `tangent_type` runs.
    Tt = tangent_type(P)
    t = randn_tangent(rng, p)
    z = zero_tangent(p)

    # Check that user-provided tangents have the same type as `tangent_type` expects.
    @test T == Tt

    # Check that ismutabletype(P) => ismutabletype(T)
    if ismutabletype(P) && !(Tt == NoTangent)
        @test ismutabletype(Tt)
    end

    # Check that tangents are of the correct type.
    @test Tt == typeof(t)
    @test Tt == typeof(z)

    # Check that zero_tangent is deterministic.
    @test has_equal_data(z, Taped.zero_tangent(p))

    # Check that zero_tangent infers.
    @test has_equal_data(z, @inferred Taped.zero_tangent(p))

    # Verify that the zero tangent is zero via its action.
    zc = deepcopy(z)
    tc = deepcopy(t)
    @test has_equal_data(@inferred(increment!!(zc, zc)), zc)
    @test has_equal_data(increment!!(zc, tc), tc)
    @test has_equal_data(increment!!(tc, zc), tc)

    if ismutabletype(P)
        @test increment!!(zc, zc) === zc
        @test increment!!(tc, zc) === tc
        @test increment!!(zc, tc) === zc
        @test increment!!(tc, tc) === tc
    end

    z_pred = increment!!(x, y)
    @test has_equal_data(z_pred, z_target)
    if ismutabletype(P)
        @test z_pred === x
    end

    # If t isn't the zero element, then adding it to itself must change its value.
    if !has_equal_data(t, z)
        if !ismutabletype(P)
            tc′ = increment!!(tc, tc)
            @test tc === tc′ || !has_equal_data(tc′, tc)
        end
    end

    # Adding things preserves types.
    @test increment!!(zc, zc) isa Tt
    @test increment!!(zc, tc) isa Tt
    @test increment!!(tc, zc) isa Tt

    # Setting to zero equals zero.
    @test has_equal_data(set_to_zero!!(tc), z)
    if ismutabletype(P)
        @test set_to_zero!!(tc) === tc
    end

    z = zero_tangent(p)
    r = randn_tangent(rng, p)

    # Verify that operations required for finite difference testing to run, and produce the
    # correct output type.
    @test _add_to_primal(p, t) isa P
    @test _diff(p, p) isa T
    @test _dot(t, t) isa Float64
    @test _scale(11.0, t) isa T
    @test populate_address_map(p, t) isa AddressMap

    # Run some basic numerical sanity checks on the output the functions required for finite
    # difference testing. These are necessary but insufficient conditions.
    @test has_equal_data(_add_to_primal(p, z), p)
    if !has_equal_data(z, r)
        @test !has_equal_data(_add_to_primal(p, r), p)
    end
    @test has_equal_data(_diff(p, p), zero_tangent(p))
    @test _dot(t, t) >= 0.0
    @test _dot(t, zero_tangent(p)) == 0.0
    @test _dot(t, increment!!(deepcopy(t), t)) ≈ 2 * _dot(t, t)
    @test has_equal_data(_scale(1.0, t), t)
    @test has_equal_data(_scale(2.0, t), increment!!(deepcopy(t), t))
end

function test_equality_comparison(x)
    @nospecialize x
    @test has_equal_data(x, x) isa Bool
    @test has_equal_data_up_to_undefs(x, x) isa Bool
    @test has_equal_data(x, x)
    @test has_equal_data_up_to_undefs(x, x)
end

function run_hand_written_rrule!!_test_cases(rng_ctor, v::Val)
    test_cases, memory = Taped.generate_hand_written_rrule!!_test_cases(rng_ctor, v)
    GC.@preserve memory @testset "$f, $(typeof(x))" for (interface_only, perf_flag, _, f, x...) in test_cases
        test_rrule!!(rng_ctor(123), f, x...; interface_only, perf_flag)
    end
end

function run_derived_rrule!!_test_cases(rng_ctor, v::Val)
    test_cases, memory = Taped.generate_derived_rrule!!_test_cases(rng_ctor, v)
    GC.@preserve memory @testset "$f, $(typeof(x))" for (interface_only, _, f, x...) in test_cases
        test_taped_rrule!!(rng_ctor(123), f, x...; interface_only)
    end
end

function run_rrule!!_test_cases(rng_ctor, v::Val)
    run_hand_written_rrule!!_test_cases(rng_ctor, v)
    run_derived_rrule!!_test_cases(rng_ctor, v)
end

end



module TestResources

using ..Taped
using ..Taped: CoDual, Tangent, MutableTangent, NoTangent, PossiblyUninitTangent

using DiffTests, LinearAlgebra, Random, Setfield


#
# Types used for testing purposes
#

function equal_field(a, b, f)
    (!isdefined(a, f) || !isdefined(b, f)) && return true
    return getfield(a, f) == getfield(b, f)
end

mutable struct Foo
    x::Real
end

Base.:(==)(a::Foo, b::Foo) = equal_field(a, b, :x)

struct StructFoo
    a::Real
    b::Vector{Float64}
    StructFoo(a::Float64, b::Vector{Float64}) = new(a, b)
    StructFoo(a::Float64) = new(a)
end

Base.:(==)(a::StructFoo, b::StructFoo) = equal_field(a, b, :a) && equal_field(a, b, :b)

mutable struct MutableFoo
    a::Float64
    b::AbstractVector
    MutableFoo(a::Float64, b::Vector{Float64}) = new(a, b)
    MutableFoo(a::Float64) = new(a)
end

Base.:(==)(a::MutableFoo, b::MutableFoo) = equal_field(a, b, :a) && equal_field(a, b, :b)

mutable struct TypeStableMutableStruct{T}
    a::Float64
    b::T
    TypeStableMutableStruct{T}(a::Float64) where {T} = new{T}(a)
    TypeStableMutableStruct{T}(a::Float64, b::T) where {T} = new{T}(a, b)
end

function Base.:(==)(a::TypeStableMutableStruct, b::TypeStableMutableStruct)
    return equal_field(a, b, :a) && equal_field(a, b, :b)
end



#
# Functions for which rules are implemented. Useful for testing basic test infrastructure,
# and ensuring that any modifications to the interface do not prevent certain functions from
# having rules written for them. If a function is found for which the design of `rrule!!`
# doesn't permit a rule to be written, it should be added here to prevent future
# regressions. For example, `primitive_setfield!` was added because a particular iteration
# of the design did not allow the implementation of a correct rule.
# The hope is that this list of functions catches any issues early, before a large-scale
# re-write of the rules begins.
#

p_sin(x) = sin(x)

function Taped.rrule!!(::CoDual{typeof(p_sin)}, x::CoDual{Float64, Float64})
    p_sin_pb!!(ȳ::Float64, df, dx) = df, dx + ȳ * cos(primal(x))
    return CoDual(sin(primal(x)), zero(Float64)), p_sin_pb!!
end

p_mul(x, y) = x * y

function Taped.rrule!!(::CoDual{typeof(p_mul)}, x::CoDual{Float64}, y::CoDual{Float64})
    p_mul_pb!!(z̄, df, dx, dy) = df, dx + z̄ * primal(y), dy + z̄ * primal(x)
    return CoDual(primal(x) * primal(y), zero(Float64)), p_mul_pb!!
end

p_mat_mul!(C, A, B) = mul!(C, A, B)

function Taped.rrule!!(
    ::CoDual{typeof(p_mat_mul!)}, C::T, A::T, B::T
) where {T<:CoDual{Matrix{Float64}}}
    C_old = copy(C)
    function p_mat_mul_pb!!(C̄::Matrix{Float64}, df, _, Ā, B̄)
        Ā .+= C̄ * primal(B)'
        B̄ .+= primal(A)' * C̄
        primal(C) .= primal(C_old)
        tangent(C) .= tangent(C_old)
        return df, C̄, Ā, B̄
    end
    mul!(primal(C), primal(A), primal(B))
    tangent(C) .= 0
    return C, p_mat_mul_pb!!
end

p_setfield!(value, name::Symbol, x) = setfield!(value, name, x)

__replace_value(::T, v) where {T<:PossiblyUninitTangent} = T(v)

function __setfield!(value::MutableTangent, name, x)
    fields = value.fields
    new_fields = @set fields.$name = __replace_value(getfield(fields, name), x)
    value.fields = new_fields
    return x
end

function Taped.rrule!!(::CoDual{typeof(p_setfield!)}, value, name::CoDual{Symbol}, x)
    _name = primal(name)
    _value = primal(value)
    _dvalue = tangent(value)
    old_x = getfield(_value, _name)
    old_dx = getfield(_dvalue.fields, _name).tangent

    function p_setfield!_pb!!(dy, df, dvalue, dname, dx)

        # Add all increments to dx.
        dx = increment!!(dx, getfield(dvalue.fields, _name).tangent)
        dx = increment!!(dx, dy)

        # Restore old values.
        setfield!(primal(value), _name, old_x)
        __setfield!(tangent(value), _name, old_dx)

        return df, dvalue, dname, dx
    end

    y = CoDual(
        setfield!(_value, _name, primal(x)),
        __setfield!(_dvalue, _name, tangent(x)),
    )
    return y, p_setfield!_pb!!
end

for f in [p_sin, p_mul, p_mat_mul!, p_setfield!]
    @eval Taped.Umlaut.isprimitive(::Taped.RMC, ::typeof($f), x...) = true
end

const __A = randn(3, 3)

const PRIMITIVE_TEST_FUNCTIONS = Any[
    (:stability, p_sin, 5.0),
    (:none, p_mat_mul!, randn(4, 5), randn(4, 3), randn(3, 5)),
    (:none, p_mul, 5.0, 4.0),
    (:none, p_mat_mul!, randn(3, 3), __A, __A),
    (:none, p_setfield!, Foo(5.0), :x, 4.0),
    (:none, p_setfield!, MutableFoo(5.0, randn(5)), :b, randn(6)),
    (:none, p_setfield!, MutableFoo(5.0), :a, 5.0),
]

#
# Tests for AD. There are not rules defined directly on these functions, and they require
# that most language primitives have rules defined.
#

test_sin(x) = sin(x)

test_cos_sin(x) = cos(sin(x))

test_isbits_multiple_usage(x::Float64) = Core.Intrinsics.mul_float(x, x)

function test_isbits_multiple_usage_2(x::Float64)
    y = Core.Intrinsics.mul_float(x, x)
    return Core.Intrinsics.mul_float(y, y)
end

function test_isbits_multiple_usage_3(x::Float64)
    y = sin(x)
    z = Core.Intrinsics.mul_float(y, y)
    a = Core.Intrinsics.mul_float(z, z)
    b = cos(a)
    return b
end

function test_isbits_multiple_usage_4(x::Float64)
    y = x > 0.0 ? cos(x) : sin(x)
    return Core.Intrinsics.mul_float(y, y)
end

function test_isbits_multiple_usage_5(x::Float64)
    y = Core.Intrinsics.mul_float(x, x)
    return x > 0.0 ? cos(y) : sin(y)
end

test_getindex(x::AbstractArray{<:Real}) = x[1]

function test_mutation!(x::AbstractVector{<:Real})
    x[1] = sin(x[2])
    return x[1]
end

function test_for_loop(x)
    for _ in 1:5
        x = sin(x)
    end
    return x
end

function test_while_loop(x)
    n = 3
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end

test_mutable_struct_basic(x) = Foo(x).x

test_mutable_struct_basic_sin(x) = sin(Foo(x).x)

function test_mutable_struct_setfield(x)
    foo = Foo(1.0)
    foo.x = x
    return foo.x
end

function test_mutable_struct(x)
    foo = Foo(x)
    foo.x = sin(foo.x)
    return foo.x
end

test_struct_partial_init(a::Float64) = StructFoo(a).a

test_mutable_partial_init(a::Float64) = MutableFoo(a).a

function test_naive_mat_mul!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}) where {T<:Real}
    for p in 1:size(C, 1)
        for q in 1:size(C, 2)
            C[p, q] = zero(T)
            for r in 1:size(A, 2)
                C[p, q] += A[p, r] * B[r, q]
            end
        end
    end
    return C
end

test_diagonal_to_matrix(D::Diagonal) = Matrix(D)

relu(x) = max(x, zero(x))

test_mlp(x, W1, W2) = W2 * relu.(W1 * x)

function generate_test_functions()
    return Any[
        (false, (lb=100, ub=10_000), test_sin, 1.0),
        (false, (lb=100, ub=10_000), test_cos_sin, 2.0),
        (false, (lb=1_000, ub=20_000), test_isbits_multiple_usage, 5.0),
        (false, (lb=1_000, ub=50_000), test_isbits_multiple_usage_2, 5.0),
        (false, (lb=1_000, ub=20_000), test_isbits_multiple_usage_3, 4.1),
        (false, (lb=1_000, ub=100_000), test_isbits_multiple_usage_4, 5.0),
        (false, (lb=1_000, ub=100_000), test_isbits_multiple_usage_5, 4.1),
        (false, (lb=1_000, ub=20_000), test_getindex, [1.0, 2.0]),
        (false, (lb=1_000, ub=50_000), test_mutation!, [1.0, 2.0]),
        (false, (lb=1_000, ub=25_000), test_while_loop, 2.0),
        (false, (lb=1_000, ub=100_000), test_for_loop, 3.0),
        (false, (lb=1_000, ub=50_000), test_mutable_struct_basic, 5.0),
        (false, (lb=1_000, ub=20_000), test_mutable_struct_basic_sin, 5.0),
        (false, (lb=1_000, ub=100_000), test_mutable_struct_setfield, 4.0),
        (false, (lb=1_000, ub=25_000), test_mutable_struct, 5.0),
        (false, (lb=1_000, ub=50_000), test_struct_partial_init, 3.5),
        (false, (lb=1_000, ub=50_000), test_mutable_partial_init, 3.3),
        (
            false,
            (lb=100_000, ub=10_000_000),
            test_naive_mat_mul!, randn(2, 1), randn(2, 1), randn(1, 1),
        ),
        (
            false,
            (lb=10_000, ub=5_000_000),
            (A, C) -> test_naive_mat_mul!(C, A, A), randn(2, 2), randn(2, 2),
        ),
        (false, (lb=10_000, ub=50_000_000), sum, randn(3)),
        (false, (lb=10_000, ub=50_000_000), test_diagonal_to_matrix, Diagonal(randn(3))),
        (
            false,
            (lb=10_000, ub=50_000_000),
            ldiv!, randn(2, 2), Diagonal(rand(2) .+ 1), randn(2, 2),
        ),
        (
            false,
            (lb=10_000, ub=50_000_000),
            kron!, randn(4, 4), Diagonal(randn(2)), randn(2, 2),
        ),
        (
            false,
            (lb=100_000, ub=100_000_000),
            test_mlp, randn(5, 2), randn(7, 5), randn(3, 7),
        ),
    ]
end

function value_dependent_control_flow(x, n)
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end

#
# This is a version of setfield! in which there is an issue with the address map.
# The method of setfield! is incorrectly implemented, so it errors. This is intentional,
# and is used to ensure that the tests correctly pick up on this mistake.
#

my_setfield!(args...) = setfield!(args...)

function _setfield!(value::MutableTangent, name, x)
    @set value.fields.$name = x
    return x
end

function Taped.rrule!!(::Taped.CoDual{typeof(my_setfield!)}, value, name, x)
    _name = primal(name)
    old_x = isdefined(primal(value), _name) ? getfield(primal(value), _name) : nothing
    function setfield!_pullback(dy, df, dvalue, ::NoTangent, dx)
        new_dx = increment!!(dx, getfield(dvalue.fields, _name).tangent)
        new_dx = increment!!(new_dx, dy)
        old_x !== nothing && setfield!(primal(value), _name, old_x)
        return df, dvalue, NoTangent(), new_dx
    end
    y = Taped.CoDual(
        setfield!(primal(value), _name, primal(x)),
        _setfield!(tangent(value), _name, tangent(x)),
    )
    return y, setfield!_pullback
end

# Tests brought in from DiffTests.jl
const _n = rand()
const _x = rand(5, 5)
const _y = rand(26)
const _A = rand(5, 5)
const _B = rand(5, 5)
const _rng = Xoshiro(123456)

const DIFFTESTS_FUNCTIONS = vcat(
    tuple.(
        fill(false, length(DiffTests.NUMBER_TO_NUMBER_FUNCS)),
        DiffTests.NUMBER_TO_NUMBER_FUNCS,
        rand(_rng, length(DiffTests.NUMBER_TO_NUMBER_FUNCS)) .+ 1e-1,
    ),
    tuple.(
        fill(false, length(DiffTests.NUMBER_TO_ARRAY_FUNCS)),
        DiffTests.NUMBER_TO_ARRAY_FUNCS,
        [rand(_rng) + 1e-1 for _ in DiffTests.NUMBER_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS)),
        DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS,
        [rand(_rng, 5) .+ 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
        [rand(_rng) + 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.VECTOR_TO_NUMBER_FUNCS)),
        DiffTests.VECTOR_TO_NUMBER_FUNCS,
        [rand(_rng, 5) .+ 1e-1 for _ in DiffTests.VECTOR_TO_NUMBER_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.MATRIX_TO_NUMBER_FUNCS)),
        DiffTests.MATRIX_TO_NUMBER_FUNCS,
        [rand(_rng, 5, 5) .+ 1e-1 for _ in DiffTests.MATRIX_TO_NUMBER_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS)),
        DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS,
        [rand(_rng, 5, 5) .+ 1e-1 + I for _ in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS],
        [rand(_rng, 5, 5) .+ 1e-1 + I for _ in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS)),
        DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS,
        [rand(_rng, 5, 5) .+ 1e-1 for _ in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS],
        [rand(_rng, 5, 5) .+ 1e-1 for _ in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS],
        [rand(_rng, 5, 5) .+ 1e-1 for _ in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS)),
        DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS,
        [rand(_rng, 26) .+ 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
        [rand(_rng, 26) .+ 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.VECTOR_TO_VECTOR_FUNCS)),
        DiffTests.VECTOR_TO_VECTOR_FUNCS,
        [rand(_rng, 26) .+ 1e-1 for _ in DiffTests.VECTOR_TO_VECTOR_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.ARRAY_TO_ARRAY_FUNCS)),
        DiffTests.ARRAY_TO_ARRAY_FUNCS,
        [rand(_rng, 26) .+ 1e-1 for _ in DiffTests.ARRAY_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.MATRIX_TO_MATRIX_FUNCS)),
        DiffTests.MATRIX_TO_MATRIX_FUNCS,
        [rand(_rng, 5, 5) .+ 1e-1 for _ in DiffTests.MATRIX_TO_MATRIX_FUNCS],
    ),
)

end
