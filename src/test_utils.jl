"""
    module TestTypes

A module containing types and associated utility functionality against which standardised
functionality can be tested. The goal is to provide sufficiently broad coverage of different
kinds of types to provide a high degree of confidence that correctness and performance
of tangent operations generalises to new types found in the wild.
"""
module TestTypes

using Base.Iterators: product
using Core: svec
using ExprTools: combinedef
using ..Tapir: NoTangent, tangent_type, _typeof

const PRIMALS = Tuple{Bool, Any, Tuple}[]

# Generate all of the composite types against which we might wish to test.
function generate_primals()
    empty!(PRIMALS)
    for n_fields in [0, 1, 2], is_mutable in [true, false]

        # Generate all possible permutations of primitive fields types.
        fields = (
            (type=Float64, primal=1.0, tangent=1.0),
            (type=Int64, primal=1, tangent=NoTangent()),
            (type=Vector{Float64}, primal=ones(2), tangent=ones(2)),
            (type=Vector{Int64}, primal=Int64[1, 1], tangent=fill(NoTangent(), 2)),
        )
        field_combinations = vec(collect(product(fill(fields, n_fields)...)))

        ns_always_def = 0:n_fields

        for fields in field_combinations, n_always_def in ns_always_def

            mutable_str = is_mutable ? "Mutable" : ""
            field_types = map(x -> x.type, fields)
            type_string = join(map(string, field_types), "_")
            name = Symbol("$(mutable_str)Struct_$(type_string)_$(n_always_def)")
            field_names = map(n -> Symbol("x$n"), 1:n_fields)

            # Create the specified type.
            struct_expr = Expr(
                :struct,
                is_mutable,
                name,
                Expr(
                    :block,

                    # Specify fields.
                    map(n -> Expr(:(::), field_names[n], field_types[n]), 1:n_fields)...,

                    # Specify inner constructors.
                    map(n_always_def:n_fields) do n
                        return combinedef(Dict(
                            :head => :function,
                            :name => name,
                            :args => field_names[1:n],
                            :body => Expr(:call, :new, field_names[1:n]...),
                        ))
                    end...,
                ),
            )
            @eval $(struct_expr)

            t = @eval $name
            for n in n_always_def:n_fields
                interface_only = any(x -> isbitstype(x.type), fields[n+1:end])
                fields_copies = map(x -> deepcopy(x.primal), fields[1:n])
                push!(PRIMALS, (interface_only, t, fields_copies))
            end
        end
    end
    return nothing
end

instantiate(test_case) = (test_case[1], test_case[2](test_case[3]...))

end

"""
    module TestUtils

A collection of functions comprising collections of unit tests which check to see if the
interfaces that this package defines have been implemented correctly.
"""
module TestUtils

using Random, Tapir, Test, InteractiveUtils
using Tapir:
    CoDual, NoTangent, rrule!!, is_init, zero_codual, DefaultCtx, @is_primitive, val,
    is_always_fully_initialised, get_tangent_field, set_tangent_field!, MutableTangent,
    Tangent, _typeof, rdata, NoFData, to_fwds, uninit_fdata, zero_rdata,
    zero_rdata_from_type, CannotProduceZeroRDataFromType, lazy_zero_rdata, instantiate,
    can_produce_zero_rdata_from_type, increment_rdata!!, fcodual_type,
    verify_fdata_type, verify_rdata_type, verify_fdata_value, verify_rdata_value,
    InvalidFDataException, InvalidRDataException

struct Shim end

function test_opt(::Any, args...)
    throw(error("Load JET to use this function."))
end

function report_opt(::Any, tt)
    throw(error("Load JET to use this function."))
end

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
function has_equal_data(x::Float64, y::Float64; equal_undefs=true)
    return (isapprox(x, y) && !isnan(x)) || (isnan(x) && isnan(y))
end
function has_equal_data(x::T, y::T; equal_undefs=true) where {T<:Core.SimpleVector}
    return all(map((a, b) -> has_equal_data(a, b; equal_undefs), x, y))
end
function has_equal_data(x::T, y::T; equal_undefs=true) where {T}
    isprimitivetype(T) && return isequal(x, y)
    if ismutabletype(x)
        return all(map(
            n -> isdefined(x, n) ? has_equal_data(getfield(x, n), getfield(y, n)) : true,
            fieldnames(T),
        ))
    else
        for n in fieldnames(T)
            if isdefined(x, n)
                if isdefined(y, n) && has_equal_data(getfield(x, n), getfield(y, n))
                    continue
                else
                    return false
                end
            else
                return isdefined(y, n) ? false : true
            end
        end
        return true
    end
end
function has_equal_data(x::GlobalRef, y::GlobalRef; equal_undefs=true)
    return x.mod == y.mod && x.name == y.name
end

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
    T === NoTangent && return m
    T === NoFData && return m
    if ismutabletype(P)
        @assert T <: MutableTangent
        k = pointer_from_objref(primal)
        v = pointer_from_objref(tangent)
        haskey(m, k) && (@assert (m[k] == v))
        m[k] = v
    end
    foreach(fieldnames(P)) do n
        t_field = __get_data_field(tangent, n)
        if isdefined(primal, n) && is_init(t_field)
            populate_address_map!(m, getfield(primal, n), val(t_field))
        elseif isdefined(primal, n) && !is_init(t_field)
            throw(error("unhandled defined-ness"))
        elseif !isdefined(primal, n) && is_init(t_field)
            throw(error("unhandled defined-ness"))
        end
    end
    return m
end

__get_data_field(t::Union{Tangent, MutableTangent}, n) = getfield(t.fields, n)
__get_data_field(t::Union{Tapir.FData, Tapir.RData}, n) = getfield(t.data, n)

function populate_address_map!(m::AddressMap, p::P, t) where {P<:Union{Tuple, NamedTuple}}
    t isa NoFData && return m
    t isa NoTangent && return m
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
function test_rrule_numerical_correctness(rng::AbstractRNG, f_f̄, x_x̄...; rule)
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
    ẋ = map(_x -> randn_tangent(rng, _x), x)
    ε = 1e-7
    x′ = _add_to_primal(x, _scale(ε, ẋ))
    y′ = f(x′...)
    ẏ = _scale(1 / ε, _diff(y′, y_primal))
    ẋ_post = map((_x′, _x_p) -> _scale(1 / ε, _diff(_x′, _x_p)), x′, x_primal)

    # Run `rrule!!` on copies of `f` and `x`. We use randomly generated tangents so that we
    # can later verify that non-zero values do not get propagated by the rule.
    x̄_zero = map(zero_tangent, x)
    x̄_fwds = map(Tapir.fdata, x̄_zero)
    x_x̄_rule = map((x, x̄_f) -> fcodual_type(_typeof(x))(_deepcopy(x), x̄_f), x, x̄_fwds)
    inputs_address_map = populate_address_map(map(primal, x_x̄_rule), map(tangent, x_x̄_rule))
    y_ȳ_rule, pb!! = rule(to_fwds(f_f̄), x_x̄_rule...)

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

    ȳ_init = set_to_zero!!(zero_tangent(primal(y_ȳ_rule), tangent(y_ȳ_rule)))
    x̄_init = map(set_to_zero!!, x̄_zero)
    ȳ = increment!!(ȳ_init, ȳ_delta)
    map(increment!!, x̄_init, x̄_delta)
    _, x̄_rvs_inc... = pb!!(Tapir.rdata(ȳ))
    x̄_rvs = map((x, x_inc) -> increment!!(rdata(x), x_inc), x̄_delta, x̄_rvs_inc)
    x̄ = map(tangent, x̄_fwds, x̄_rvs)

    # Check that inputs have been returned to their original value.
    @test all(map(has_equal_data_up_to_undefs, x, map(primal, x_x̄_rule)))

    # pullbacks increment, so have to compare to the incremented quantity.
    @test _dot(ȳ_delta, ẏ) + _dot(x̄_delta, ẋ_post) ≈ _dot(x̄, ẋ) rtol=1e-3 atol=1e-3
end

get_address(x) = ismutable(x) ? pointer_from_objref(x) : nothing

_deepcopy(x) = deepcopy(x)
_deepcopy(x::Module) = x

rrule_output_type(::Type{Ty}) where {Ty} = Tuple{Tapir.fcodual_type(Ty), Any}

function test_rrule_interface(f_f̄, x_x̄...; rule)
    @nospecialize f_f̄ x_x̄

    # Pull out primals and run primal computation.
    f = primal(f_f̄)
    f̄ = tangent(f_f̄)
    x_x̄ = map(_deepcopy, x_x̄)
    x = map(primal, x_x̄)
    x̄ = map(tangent, x_x̄)

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

    # Extract the forwards-data from the tangents.
    f_fwds = to_fwds(f_f̄)
    x_fwds = map(to_fwds, x_x̄)

    # Run the rrule, check it has output a thing of the correct type, and extract results.
    # Throw a meaningful exception if the rrule doesn't run at all.
    x_addresses = map(get_address, x)
    rrule_ret = try
        rule(f_fwds, x_fwds...)
    catch e
        display(e)
        println()
        throw(ArgumentError(
            "rrule!! for $(_typeof(f_fwds)) with argument types $(_typeof(x_fwds)) does not run."
        ))
    end
    @test rrule_ret isa rrule_output_type(_typeof(y))
    y_ȳ, pb!! = rrule_ret

    # Run the reverse-pass. Throw a meaningful exception if it doesn't run at all.
    ȳ = Tapir.rdata(zero_tangent(primal(y_ȳ), tangent(y_ȳ)))
    f̄_new, x̄_new... = try
        pb!!(ȳ)
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
    @test _typeof(f̄_new) == _typeof(rdata(f̄))
    @test all(map((a, b) -> _typeof(a) == _typeof(rdata(b)), x̄_new, x̄))
end

function __forwards_and_backwards(rule, x_x̄::Vararg{Any, N}) where {N}
    out, pb!! = rule(x_x̄...)
    return pb!!(Tapir.zero_rdata(primal(out)))
end

function test_rrule_performance(
    performance_checks_flag::Symbol, rule::R, f_f̄::F, x_x̄::Vararg{Any, N}
) where {R, F, N}

    # Verify that a valid performance flag has been passed.
    valid_flags = (:none, :stability, :allocs, :stability_and_allocs)
    if !in(performance_checks_flag, valid_flags)
        throw(ArgumentError(
            "performance_checks=$performance_checks_flag. Must be one of $valid_flags"
        ))
    end
    performance_checks_flag == :none && return nothing

    if performance_checks_flag in (:stability, :stability_and_allocs)

        # Test primal stability.
        test_opt(Shim(), primal(f_f̄), map(_typeof ∘ primal, x_x̄))

        # Test forwards-pass stability.
        test_opt(Shim(), rule, (_typeof(to_fwds(f_f̄)), map(_typeof ∘ to_fwds, x_x̄)...))

        # Test reverse-pass stability.
        y_ȳ, pb!! = rule(to_fwds(f_f̄), map(to_fwds, _deepcopy(x_x̄))...)
        rvs_data = Tapir.rdata(zero_tangent(primal(y_ȳ), tangent(y_ȳ)))
        test_opt(Shim(), pb!!, (_typeof(rvs_data), ))
    end

    if performance_checks_flag in (:allocs, :stability_and_allocs)
        f = primal(f_f̄)
        x = map(primal, x_x̄)

        # Test allocations in primal.
        f(x...)
        @test (@allocations f(x...)) == 0

        # Test allocations in round-trip.
        f_f̄_fwds = to_fwds(f_f̄)
        x_x̄_fwds = map(to_fwds, x_x̄)
        __forwards_and_backwards(rule, f_f̄_fwds, x_x̄_fwds...)
        @test (@allocations __forwards_and_backwards(rule, f_f̄_fwds, x_x̄_fwds...)) == 0
    end
end

@doc"""
    test_rule(
        rng, x...;
        interface_only=false,
        has_rrule!!=true,
        perf_flag::Symbol,
        ctx=DefaultCtx(),
        safety_on=false,
    )

Run standardised tests on the `rule` for `x`.
The first element of `x` should be the primal function to test, and each other element a
positional argument.
In most cases, elements of `x` can just be the primal values, and `randn_tangent` can be
relied upon to generate an appropriate tangent to test. Some notable exceptions exist
though, in partcular `Ptr`s. In this case, the argument for which `randn_tangent` cannot be
readily defined should be a `CoDual` containing the primal, and a _manually_ constructed
tangent field.

This function uses [`Tapir.build_rrule`](@ref) to construct a rule. This will use an
`rrule!!` if one exists, and derive a rule otherwise.
"""
function test_rule(
    rng, x...;
    interface_only::Bool=false,
    is_primitive::Bool=true,
    perf_flag::Symbol,
    interp::Tapir.TapirInterpreter,
    safety_on::Bool=false,
)
    @nospecialize rng x

    # Construct the rule.
    rule = Tapir.build_rrule(interp, _typeof(__get_primals(x)); safety_on)

    # If we're requiring `is_primitive`, then check that `rule == rrule!!`.
    if is_primitive
        @test rule === rrule!!
    end

    # Generate random tangents for anything that is not already a CoDual.
    x_x̄ = map(x -> x isa CoDual ? x : zero_codual(x), x)

    # Test that the interface is basically satisfied (checks types / memory addresses).
    test_rrule_interface(x_x̄...; rule)

    # Test that answers are numerically correct / consistent.
    interface_only || test_rrule_numerical_correctness(rng, x_x̄...; rule)

    # Test the performance of the rule.
    test_rrule_performance(perf_flag, rule, x_x̄...)
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
    names = filter(f -> isdefined(x, f), fieldnames(_typeof(x)))
    return map(n -> (x, n), vcat(names..., eachindex(names)...))
end
generate_args(::typeof(isa), x) = [(x, Float64), (x, Int), (x, _typeof(x))]
function generate_args(::typeof(setfield!), x)
    names = filter(f -> isdefined(x, f), fieldnames(_typeof(x)))
    return map(n -> (x, n, getfield(x, n)), vcat(names..., eachindex(names)...))
end
generate_args(::typeof(tuple), x) = [(x, ), (x, x), (x, x, x)]
generate_args(::typeof(typeassert), x) = [(x, _typeof(x))]
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
            test_rule(
                rng, f, args...;
                interface_only=false,
                is_primitive=true,
                perf_flag=:none,
                interp=Tapir.TapirInterpreter(),
            )
        end
    end
end

"""
    test_tangent_consistency(rng::AbstractRNG, p::P; interface_only=false) where {P}

Like `test_tangent`, but relies on `zero_tangent` and `randn_tangent` to generate test
cases. Consequently, it is not possible to verify that `increment!!` produces the correct
numbers in an absolute sense, only that all operations are self-consistent and have the
performance one would expect of them.

Setting `interface_only` to `true` turns off all numerical correctness checks. This is
useful when `p` contains uninitialised isbits data, whose value is non-deterministic.
This happens for `Array`s of isbits data, and in composite types with uninitialised isbits
fields. In such situations, it still makes sense to test the performance of tangent
generation and incrementation operations, but owing to the non-determinism it makes no sense
to check their numerical correctness. 
"""
function test_tangent_consistency(rng::AbstractRNG, p::P; interface_only=false) where {P}
    @nospecialize rng p

    # Test that basic interface works.
    T = tangent_type(P)
    @test T isa Type
    z = zero_tangent(p)
    @test z isa T
    t = randn_tangent(rng, p)
    @test t isa T
    test_equality_comparison(p)
    test_equality_comparison(t)

    # Check that zero_tangent isn't obviously non-deterministic.
    @test has_equal_data(z, Tapir.zero_tangent(p))

    # Check that ismutabletype(P) => ismutabletype(T)
    if ismutabletype(P) && !(T == NoTangent)
        @test ismutabletype(T)
    end

    # Verify z is zero via its action on t.
    zc = deepcopy(z)
    tc = deepcopy(t)
    @test has_equal_data(@inferred(increment!!(zc, zc)), zc)
    @test has_equal_data(increment!!(zc, tc), tc)
    @test has_equal_data(increment!!(tc, zc), tc)

    # increment!! preserves types.
    @test increment!!(zc, zc) isa T
    @test increment!!(zc, tc) isa T
    @test increment!!(tc, zc) isa T

    # The output of `increment!!` for a mutable type must have the property that the first
    # argument === the returned value.
    if ismutabletype(P)
        @test increment!!(zc, zc) === zc
        @test increment!!(tc, zc) === tc
        @test increment!!(zc, tc) === zc
        @test increment!!(tc, tc) === tc
    end

    # If t isn't the zero element, then adding it to itself must change its value.
    if !has_equal_data(t, z) && !ismutabletype(P)
        tc′ = increment!!(tc, tc)
        @test tc === tc′ || !has_equal_data(tc′, tc)
    end

    # Setting to zero equals zero.
    @test has_equal_data(set_to_zero!!(tc), z)
    if ismutabletype(P)
        @test set_to_zero!!(tc) === tc
    end

    z = zero_tangent(p)
    r = randn_tangent(rng, p)

    # Check set_tangent_field if mutable.
    t isa MutableTangent && test_set_tangent_field!_correctness(t, z)

    # Verify that operations required for finite difference testing to run, and produce the
    # correct output type.
    @test _add_to_primal(p, t) isa P
    @test _diff(p, p) isa T
    @test _dot(t, t) isa Float64
    @test _scale(11.0, t) isa T
    @test populate_address_map(p, t) isa AddressMap

    # Run some basic numerical sanity checks on the output the functions required for finite
    # difference testing. These are necessary but insufficient conditions.
    if !interface_only
        @test has_equal_data(_add_to_primal(p, z), p)
        if !has_equal_data(z, r)
            @test !has_equal_data(_add_to_primal(p, r), p)
        end
        @test has_equal_data(_diff(p, p), zero_tangent(p))
    end
    @test _dot(t, t) >= 0.0
    @test _dot(t, zero_tangent(p)) == 0.0
    @test _dot(t, increment!!(deepcopy(t), t)) ≈ 2 * _dot(t, t)
    @test has_equal_data(_scale(1.0, t), t)
    @test has_equal_data(_scale(2.0, t), increment!!(deepcopy(t), t))
end

function test_set_tangent_field!_correctness(t1::T, t2::T) where {T<:MutableTangent}
    Tfields = _typeof(t1.fields)
    for n in 1:fieldcount(Tfields)
        !Tapir.is_init(t2.fields[n]) && continue
        v = get_tangent_field(t2, n)

        # Int form.
        v′ = Tapir.set_tangent_field!(t1, n, v)
        @test v′ === v
        @test Tapir.get_tangent_field(t1, n) === v

        # Symbol form.
        s = fieldname(Tfields, n)
        g = Tapir.set_tangent_field!(t1, s, v)
        @test g === v
        @test Tapir.get_tangent_field(t1, n) === v
    end
end

"""
    test_tangent_performance(rng::AbstractRNG, p::P) where {P}

Runs a variety of performance-related tests on tangents. These tests constitute a set of
necessary conditions for good overall performance.

The performance model in a few cases is a little bit complicated, because it depends on
various properties of the type in question (is it mutable, are its fields mutable, are all
of its fields necessarily defined, etc), so the source code should be consulted for precise
details.

*Note:* this function assumes that the tangent interface is implemented correctly for `p`.
To verify that this is the case, ensure that all tests in either `test_tangent` or
`test_tangent_consistency` pass.
"""
function test_tangent_performance(rng::AbstractRNG, p::P) where {P}

    # Should definitely infer, because tangent type must be known statically from primal.
    z = @inferred zero_tangent(p)
    t = @inferred randn_tangent(rng, p)

    # Computing the tangent type must always be type stable and allocation-free.
    @inferred tangent_type(P)
    @test (@allocations tangent_type(P)) == 0

    # Check there are no allocations when there ought not to be.
    if !__tangent_generation_should_allocate(P)
        test_opt(Shim(), Tuple{typeof(zero_tangent), P})
        test_opt(Shim(), Tuple{typeof(randn_tangent), Xoshiro, P})
    end

    # `increment!!` should always infer.
    @inferred increment!!(t, z)
    @inferred increment!!(z, t)
    @inferred increment!!(t, t)
    @inferred increment!!(z, z)

    # Unfortunately, `increment!!` does occassionally allocate at the minute due to the
    # way we're handling partial initialisation. Hopefully this will change in the future.
    if !__increment_should_allocate(P)
        @test (@allocations increment!!(t, t)) == 0
        @test (@allocations increment!!(z, t)) == 0
        @test (@allocations increment!!(t, z)) == 0
        @test (@allocations increment!!(z, z)) == 0
    end

    # set_tangent_field! should never allocate.
    t isa MutableTangent && test_set_tangent_field!_performance(t, z)
    t isa Union{MutableTangent, Tangent} && test_get_tangent_field_performance(t)
end

_set_tangent_field!(x, ::Val{i}, v) where {i} = set_tangent_field!(x, i, v)
_get_tangent_field(x, ::Val{i}) where {i} = get_tangent_field(x, i)

function test_set_tangent_field!_performance(t1::T, t2::T) where {V, T<:MutableTangent{V}}
    for n in 1:fieldcount(V)
        !is_init(t2.fields[n]) && continue
        v = get_tangent_field(t2, n)

        # Int mode.
        _set_tangent_field!(t1, Val(n), v)
        report_opt(
            Shim(),
            Tuple{typeof(_set_tangent_field!), typeof(t1), Val{n}, typeof(v)},
        )

        if all(n -> !(fieldtype(V, n) <: Tapir.PossiblyUninitTangent), 1:fieldcount(V))
            i = Val(n)
            _set_tangent_field!(t1, i, v)
            @test count_allocs(_set_tangent_field!, t1, i, v) == 0
        end

        # Symbol mode.
        s = Val(fieldname(V, n))
        @inferred _set_tangent_field!(t1, s, v)
        report_opt(
            Shim(),
            Tuple{typeof(_set_tangent_field!), typeof(t1), typeof(s), typeof(v)},
        )

        if all(n -> !(fieldtype(V, n) <: Tapir.PossiblyUninitTangent), 1:fieldcount(V))
            _set_tangent_field!(t1, s, v)
            @test count_allocs(_set_tangent_field!, t1, s, v) == 0
        end
    end
end

function test_get_tangent_field_performance(t::Union{MutableTangent, Tangent})
    V = Tapir._typeof(t.fields)
    for n in 1:fieldcount(V)
        !is_init(t.fields[n]) && continue
        Tfield = fieldtype(Tapir.fields_type(Tapir._typeof(t)), n)
        !__is_completely_stable_type(Tfield) && continue

        # Int mode.
        i = Val(n)
        report_opt(Shim(), Tuple{typeof(_get_tangent_field), typeof(t), typeof(i)})
        @inferred _get_tangent_field(t, i)
        @test count_allocs(_get_tangent_field, t, i) == 0

        # Symbol mode.
        s = Val(fieldname(V, n))
        report_opt(Shim(), Tuple{typeof(_get_tangent_field), typeof(t), typeof(s)})
        @inferred _get_tangent_field(t, s)
        @test count_allocs(_get_tangent_field, t, s) == 0
    end
end

# Function barrier to ensure inference in value types.
function count_allocs(f::F, x::Vararg{Any, N}) where {F, N}
    @allocations f(x...)
end

# Returns true if both `zero_tangent` and `randn_tangent` should allocate when run on
# an object of type `P`.
function __tangent_generation_should_allocate(::Type{P}) where {P}
    (!isconcretetype(P) || isabstracttype(P)) && return true
    (fieldcount(P) == 0 && !ismutabletype(P)) && return false
    return ismutabletype(P) || any(__tangent_generation_should_allocate, fieldtypes(P))
end

__tangent_generation_should_allocate(::Type{P}) where {P<:Array} = true

function __increment_should_allocate(::Type{P}) where {P}
    return any(eachindex(fieldtypes(P))) do n
        Tapir.tangent_field_type(P, n) <: PossiblyUninitTangent
    end
end
__increment_should_allocate(::Type{Core.SimpleVector}) = true

function __is_completely_stable_type(::Type{P}) where {P}
    (!isconcretetype(P) || isabstracttype(P)) && return false
    isprimitivetype(P) && return true
    return all(__is_completely_stable_type, fieldtypes(P))
end

@doc"""
    test_tangent(rng::AbstractRNG, p::P, x::T, y::T, z_target::T) where {P, T}

Verify that primal `p` with tangents `z_target`, `x`, and `y`, satisfies the tangent
interface. If these tests pass, then it should be possible to write rules for primals
of type `P`, and to test them using [`test_rule`](@ref).

As always, there are limits to the errors that these tests can identify -- they form
necessary but not sufficient conditions for the correctness of your code.
"""
function test_tangent(
    rng::AbstractRNG, p::P, x::T, y::T, z_target::T; interface_only, perf=true
) where {P, T}
    @nospecialize rng p x y z_target

    # Check the interface.
    test_tangent_consistency(rng, p; interface_only=false)

    # Is the tangent_type of `P` what we expected?
    @test tangent_type(P) == T

    # Check that zero_tangent infers.
    @inferred Tapir.zero_tangent(p)

    # Verify that adding together `x` and `y` gives the value the user expected.
    z_pred = increment!!(x, y)
    @test has_equal_data(z_pred, z_target)
    if ismutabletype(P)
        @test z_pred === x
    end

    # Check performance is as expected.
    perf && test_tangent_performance(rng, p)
end

function test_tangent(rng::AbstractRNG, p::P; interface_only=false, perf=true) where {P}
    test_tangent_consistency(rng, p; interface_only)
    perf && test_tangent_performance(rng, p)
end

function test_equality_comparison(x)
    @nospecialize x
    @test has_equal_data(x, x) isa Bool
    @test has_equal_data_up_to_undefs(x, x) isa Bool
    @test has_equal_data(x, x)
    @test has_equal_data_up_to_undefs(x, x)
end

"""
    test_fwds_rvs_data(rng::AbstractRNG, p::P) where {P}

Verify that the forwards data and reverse data functionality associated to primal `p` works
correctly.
"""
function test_fwds_rvs_data(rng::AbstractRNG, p::P) where {P}

    # Check that fdata_type and rdata_type run and produce types.
    T = tangent_type(P)
    F = Tapir.fdata_type(T)
    @test F isa Type
    R = Tapir.rdata_type(T)
    @test R isa Type

    # Check that fdata and rdata produce the correct types.
    t = randn_tangent(rng, p)
    f = Tapir.fdata(t)
    @test f isa F
    r = Tapir.rdata(t)
    @test r isa R

    # Check that fdata / rdata validation functionality doesn't error on valid fdata / rdata
    # and does error on obviously wrong fdata / rdata.
    @test verify_fdata_type(P, F) === nothing
    @test verify_rdata_type(P, R) === nothing
    @test verify_fdata_value(p, f) === nothing
    @test verify_rdata_value(p, r) === nothing
    @test_throws InvalidFDataException verify_fdata_type(P, Int)
    @test_throws InvalidRDataException verify_rdata_type(P, Int)
    @test_throws InvalidFDataException verify_fdata_value(p, 0)
    @test_throws InvalidRDataException verify_rdata_value(p, 0)

    # Check that uninit_fdata yields data of the correct type.
    @test uninit_fdata(p) isa F

    # Compute the tangent type associated to `F` and `R`, and check it is equal to `T`.
    @test tangent_type(F, R) == T

    # Check that combining f and r yields a tangent of the correct type and value.
    t_combined = Tapir.tangent(f, r)
    @test t_combined isa T
    @test t_combined === t

    # Check that pulling out `f` and `r` from `t_combined` yields the correct values.
    @test Tapir.fdata(t_combined) === f
    @test Tapir.rdata(t_combined) === r

    # Test that `zero_rdata` produces valid reverse data.
    @test zero_rdata(p) isa R

    # Check that constructing a zero tangent from reverse data yields the original tangent.
    z = zero_tangent(p)
    f_z = Tapir.fdata(z)
    @test f_z isa Tapir.fdata_type(T)
    z_new = zero_tangent(p, f_z)
    @test z_new isa tangent_type(P)
    @test z_new === z

    # Query whether or not the rdata type can be built given only the primal type.
    can_make_zero = @inferred can_produce_zero_rdata_from_type(P)

    # Check that when the zero element is asked from the primal type alone, the result is
    # either an instance of R _or_ a `CannotProduceZeroRDataFromType`.
    test_opt(Shim(), zero_rdata_from_type, Tuple{Type{P}})
    rzero_from_type = @inferred zero_rdata_from_type(P)
    @test rzero_from_type isa R || rzero_from_type isa CannotProduceZeroRDataFromType
    @test can_make_zero != isa(rzero_from_type, CannotProduceZeroRDataFromType)

    # Check that we can produce a lazy zero rdata, and that it has the correct type.
    test_opt(Shim(), lazy_zero_rdata, Tuple{P})
    lazy_rzero = @inferred lazy_zero_rdata(p)
    @test instantiate(lazy_rzero) isa R

    # Check incrementing the rdata component of a tangent yields the correct type.
    @test increment_rdata!!(t, r) isa T
end

function run_hand_written_rrule!!_test_cases(rng_ctor, v::Val)
    test_cases, memory = Tapir.generate_hand_written_rrule!!_test_cases(rng_ctor, v)
    interp = Tapir.PInterp()
    GC.@preserve memory @testset "$f, $(_typeof(x))" for (interface_only, perf_flag, _, f, x...) in test_cases
        test_rule(
            rng_ctor(123), f, x...; interface_only, perf_flag, interp, safety_on=false
        )
    end
end

function run_derived_rrule!!_test_cases(rng_ctor, v::Val)
    interp = Tapir.PInterp()
    test_cases, memory = Tapir.generate_derived_rrule!!_test_cases(rng_ctor, v)
    GC.@preserve memory @testset "$f, $(typeof(x))" for
        (interface_only, perf_flag, _, f, x...) in test_cases
        test_rule(
            rng_ctor(123), f, x...;
            interp, interface_only, perf_flag, is_primitive=false, safety_on=false,
        )
    end
end

function run_rrule!!_test_cases(rng_ctor, v::Val)
    run_hand_written_rrule!!_test_cases(rng_ctor, v)
    run_derived_rrule!!_test_cases(rng_ctor, v)
end

function to_benchmark(__rrule!!::R, dx::Vararg{CoDual, N}) where {R, N}
    dx_f = Tapir.tuple_map(x -> CoDual(primal(x), Tapir.fdata(tangent(x))), dx)
    out, pb!! = __rrule!!(dx_f...)
    return pb!!(Tapir.zero_rdata(primal(out)))
end

__get_primals(xs) = map(x -> x isa CoDual ? primal(x) : x, xs)

end

"""
    module TestResources

A collection of functions and types which should be tested. The intent is to get this module
to a state in which if we can successfully AD everything in it, we know we can successfully
AD anything.
"""
module TestResources

using ..Tapir
using ..Tapir:
    CoDual, Tangent, MutableTangent, NoTangent, PossiblyUninitTangent, ircode,
    @is_primitive, MinimalCtx, val

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

mutable struct NonDifferentiableFoo
    x::Int
    y::Bool
end

mutable struct TypeStableMutableStruct{T}
    a::Float64
    b::T
    TypeStableMutableStruct{T}(a::Float64) where {T} = new{T}(a)
    TypeStableMutableStruct{T}(a::Float64, b::T) where {T} = new{T}(a, b)
end

function Base.:(==)(a::TypeStableMutableStruct, b::TypeStableMutableStruct)
    return equal_field(a, b, :a) && equal_field(a, b, :b)
end

mutable struct TypeUnstableMutableStruct
    a::Float64
    b
end

function Base.:(==)(a::TypeUnstableMutableStruct, b::TypeUnstableMutableStruct)
    return equal_field(a, b, :a) && equal_field(a, b, :b)
end

struct TypeStableStruct{T}
    a::Int
    b::T
    TypeStableStruct{T}(a::Float64) where {T} = new{T}(a)
    TypeStableStruct{T}(a::Float64, b::T) where {T} = new{T}(a, b)
end

function Base.:(==)(a::TypeStableStruct, b::TypeStableStruct)
    return equal_field(a, b, :a) && equal_field(a, b, :b)
end

struct TypeUnstableStruct
    a::Float64
    b
end

function Base.:(==)(a::TypeUnstableStruct, b::TypeUnstableStruct)
    return equal_field(a, b, :a) && equal_field(a, b, :b)
end

mutable struct FullyInitMutableStruct
    x::Float64
    y::Vector{Float64}
end

function Base.:(==)(a::FullyInitMutableStruct, b::FullyInitMutableStruct)
    return equal_field(a, b, :x) && equal_field(a, b, :y)
end

struct StructNoFwds
    x::Float64
end

struct StructNoRvs
    x::Vector{Float64}
end

#
# Tests for AD. There are not rules defined directly on these functions, and they require
# that most language primitives have rules defined.
#

@noinline function foo(x)
    y = sin(x)
    z = cos(y)
    return z
end

# A function in which everything is non-differentiable and has no branching. Ideally, the
# reverse-pass of this function would be a no-op, and there would be no use of the block
# stack anywhere.
function non_differentiable_foo(x::Int)
    y = 5x
    z = y + x
    return 10z
end

function bar(x, y)
    x1 = sin(x)
    x2 = cos(y)
    x3 = foo(x2)
    x4 = foo(x3)
    x5 = x2 + x4
    return x5
end

function unused_expression(x, n)
    y = getfield((Float64, ), n)
    return x
end

const_tester_non_differentiable() = 1

const_tester() = cos(5.0)

intrinsic_tester(x) = 5x

function goto_tester(x)
    if x > cos(x)
        @goto aha
    end
    x = sin(x)
    @label aha
    return cos(x)
end

struct StableFoo
    x::Float64
    y::Symbol
end

new_tester(x, y) = StableFoo(x, y)

new_tester_2(x) = StableFoo(x, :symbol)

@eval function new_tester_3(x::Ref{Any})
    y = x[]
    $(Expr(:new, :y, 5.0))
end

@eval splatnew_tester(x::Ref{Tuple}) = $(Expr(:splatnew, StableFoo, :(x[])))

type_stable_getfield_tester_1(x::StableFoo) = x.x
type_stable_getfield_tester_2(x::StableFoo) = x.y

const __x_for_gref_test = 5.0
@eval globalref_tester() = $(GlobalRef(@__MODULE__, :__x_for_gref_test))

const __y_for_gref_test = false
@eval globalref_tester_bool() = $(GlobalRef(@__MODULE__, :__y_for_gref_test))

function globalref_tester_2(use_gref::Bool)
    v = use_gref ? __x_for_gref_test : 1
    return sin(v)
end

const __x_for_gref_tester_3 = 5.0
@eval globalref_tester_3() = $(GlobalRef(@__MODULE__, :__x_for_gref_tester_3))

const __x_for_gref_tester_4::Float64 = 3.0
@eval globalref_tester_4() = $(GlobalRef(@__MODULE__, :__x_for_gref_tester_4))

type_unstable_tester_0(x::Ref{Any}) = x[]

type_unstable_tester(x::Ref{Any}) = cos(x[])

type_unstable_tester_2(x::Ref{Real}) = cos(x[])

type_unstable_tester_3(x::Ref{Any}) = Foo(x[])

type_unstable_function_eval(f::Ref{Any}, x::Float64) = f[](x)

type_unstable_argument_eval(@nospecialize(f), x::Float64) = f(x)

abstractly_typed_unused_container(::StructFoo, x::Float64) = 5x

function phi_const_bool_tester(x)
    if x > 0
        a = true
    else
        a = false
    end
    return cos(a)
end

function phi_node_with_undefined_value(x::Bool, y::Float64)
    if x
        v = sin(y)
    end
    z = cos(y)
    if x
        z += v
    end
    return z
end

function pi_node_tester(y::Ref{Any})
    x = y[]
    return isa(x, Int) ? sin(x) : x
end

Base.@nospecializeinfer arg_in_pi_node(@nospecialize(x)) = x isa Bool ? x : false

function avoid_throwing_path_tester(x)
    if x < 0
        Base.throw_boundserror(1:5, 6)
    end
    return sin(x)
end

simple_foreigncall_tester(x) = ccall(:jl_array_isassigned, Cint, (Any, UInt), x, 1)

function simple_foreigncall_tester_2(a::Array{T, M}, dims::NTuple{N, Int}) where {T,N,M}
    ccall(:jl_reshape_array, Array{T,N}, (Any, Any, Any), Array{T,N}, a, dims)
end

function foreigncall_tester(x)
    return ccall(:jl_array_isassigned, Cint, (Any, UInt), x, 1) == 1 ? cos(x[1]) : sin(x[1])
end

function no_primitive_inlining_tester(x)
    X = Matrix{Float64}(undef, 5, 5) # contains a foreigncall which should never be hit
    for n in eachindex(X)
        X[n] = x
    end
    return X
end

@noinline varargs_tester(x::Vararg{Any, N}) where {N} = x

varargs_tester_2(x) = varargs_tester(x)
varargs_tester_2(x, y) = varargs_tester(x, y)
varargs_tester_2(x, y, z) = varargs_tester(x, y, z)

@noinline varargs_tester_3(x, y::Vararg{Any, N}) where {N} = sin(x), y

varargs_tester_4(x) = varargs_tester_3(x...)
varargs_tester_4(x, y) = varargs_tester_3(x...)
varargs_tester_4(x, y, z) = varargs_tester_3(x...)

splatting_tester(x) = varargs_tester(x...)
unstable_splatting_tester(x::Ref{Any}) = varargs_tester(x[]...)

function inferred_const_tester(x::Base.RefValue{Any})
    y = x[]
    y === nothing && return y
    return 5y
end
inferred_const_tester(x::Int) = x == 5 ? x : 5x

getfield_tester(x::Tuple) = x[1]
getfield_tester_2(x::Tuple) = getfield(x, 1)

function setfield_tester_left!(x::FullyInitMutableStruct, new_field)
    x.x = new_field
    return new_field
end

function setfield_tester_right!(x::FullyInitMutableStruct, new_field)
    x.y = new_field
    return new_field
end

function datatype_slot_tester(n::Int)
    return (Float64, Int)[n]
end

@noinline test_sin(x) = sin(x)

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

function test_isbits_multiple_usage_phi(x::Bool, y::Float64)
    z = x ? y : 1.0
    return z * y
end

function test_multiple_call_non_primitive(x::Float64)
    for _ in 1:2
        x = test_sin(x)
    end
    return x
end

test_getindex(x::AbstractArray{<:Real}) = x[1]

function test_mutation!(x::AbstractVector{<:Real})
    x[1] = sin(x[2])
    return x[1]
end

function test_while_loop(x)
    n = 3
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end

function test_for_loop(x)
    for _ in 1:500
        x = sin(x)
    end
    return x
end

# This catches the case where there are multiple phi nodes at the start of the block, and
# they refer to one another. It is in this instance that the distinction between phi nodes
# acting "instanteneously" and "in sequence" becomes apparent.
function test_multiple_phinode_block(x::Float64, N::Int)
    a = 1.0
    b = x
    i = 1
    while i < N
        temp = a
        a = b
        b = 2temp
        i += 1
    end
    return (a, b)
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

test_mlp(x, W1, W2) = W2 * tanh.(W1 * x)

function test_multiple_pi_nodes(x::Base.RefValue{Any})
    v = x[]
    return (v::Float64, v::Float64) # PiNode applied to the same SSAValue
end

function test_multi_use_pi_node(x::Base.RefValue{Any})
    v = x[]
    for _ in 1:2
        v = sin(v)::Float64
    end
    return v
end

function test_union_of_arrays(x::Vector{Float64}, b::Bool)
    y = randn(Xoshiro(1), Float32, 4)
    z = b ? x : y
    return 2z
end

function test_union_of_types(x::Ref{Union{Type{Float64}, Type{Int}}})
    return x[]
end

function test_small_union(x::Ref{Union{Float64, Vector{Float64}}})
    v = x[]
    return v isa Float64 ? v : v[1]
end

# Only one of these is a primitive. Lots of methods to prevent the compiler from
# over-specialising.
@noinline edge_case_tester(x::Float64) = 5x
@noinline edge_case_tester(x::Any) = 5.0
@noinline edge_case_tester(x::Float32) = 6.0
@noinline edge_case_tester(x::Int) = 10
@noinline edge_case_tester(x::String) = "hi"
@is_primitive MinimalCtx Tuple{typeof(edge_case_tester), Float64}
function Tapir.rrule!!(::CoDual{typeof(edge_case_tester)}, x::CoDual{Float64})
    edge_case_tester_pb!!(dy) = Tapir.NoRData(), 5 * dy
    return Tapir.zero_fcodual(5 * primal(x)), edge_case_tester_pb!!
end

# To test the edge case properly, call this with x = Any[5.0, false]
function test_primitive_dynamic_dispatch(x::Vector{Any})
    i = 0
    y = 0.0
    while i < 2
        i += 1
        y += edge_case_tester(x[i])
    end
    return y
end

sr(n) = Xoshiro(n)

@noinline function test_self_reference(a, b)
    return a < b ? a * b : test_self_reference(b, a) + a
end

# See https://github.com/withbayes/Tapir.jl/pull/84 for info
@noinline function test_recursive_sum(x::Vector{Float64})
    isempty(x) && return 0.0
    return @inbounds x[1] + test_recursive_sum(x[2:end])
end

# Copied over from https://github.com/TuringLang/Turing.jl/issues/1140
function _sum(x)
    z = 0
    for i in eachindex(x)
        z += x[i]
    end
    return z
end

function test_handwritten_sum(x::AbstractArray{<:Real})
    y = 0.0
    n = 0
    @inbounds while n < length(x)
        n += 1
        y += x[n]
    end
    return y
end

function test_map(x::Vector{Float64}, y::Vector{Float64})
    return map((x, y) -> sin(cos(exp(x)) + exp(y) * sin(y)), x, y)
end

test_getfield_of_tuple_of_types(n::Int) = getfield((Float64, Float64), n)

test_for_invoke(x) = 5x

inlinable_invoke_call(x::Float64) = invoke(test_for_invoke, Tuple{Float64}, x)

vararg_test_for_invoke(n::Tuple{Int, Int}, x...) = sum(x) + n[1]

function inlinable_vararg_invoke_call(
    rows::Tuple{Vararg{Int}}, n1::N, ns::Vararg{N}
) where {N}
    return invoke(vararg_test_for_invoke, Tuple{typeof(rows), Vararg{N}}, rows, n1, ns...)
end

# build_rrule should error for this function, because it references a non-isbits global ref.
const __x_for_mutable_global_ref = Ref(1.0)
function mutable_global_ref(y::Float64)
    __x_for_mutable_global_ref[] = y
    return __x_for_mutable_global_ref[]
end

# build_rrule should error for this function, because it references a non-const global ref.
__x_for_non_const_global_ref::Float64 = 5.0
function non_const_global_ref(y::Float64)
    global __x_for_non_const_global_ref = y
    return __x_for_non_const_global_ref
end

function generate_test_functions()
    return Any[
        (false, :allocs, nothing, const_tester),
        (false, :allocs, nothing, const_tester_non_differentiable),
        (false, :allocs, nothing, identity, 5.0),
        (false, :allocs, nothing, foo, 5.0),
        (false, :allocs, nothing, non_differentiable_foo, 5),
        (false, :allocs, nothing, bar, 5.0, 4.0),
        (false, :allocs, nothing, unused_expression, 5.0, 1),
        (false, :none, nothing, type_unstable_argument_eval, sin, 5.0),
        (
            false, :none, nothing,
            abstractly_typed_unused_container, StructFoo(5.0, [4.0]), 5.0,
        ),
        (false, :none, (lb=1, ub=1_000), pi_node_tester, Ref{Any}(5.0)),
        (false, :none, (lb=1, ub=1_000), pi_node_tester, Ref{Any}(5)),
        (false, :none, nothing, arg_in_pi_node, false),
        (false, :allocs, nothing, intrinsic_tester, 5.0),
        (false, :allocs, nothing, goto_tester, 5.0),
        (false, :allocs, nothing, new_tester, 5.0, :hello),
        (false, :allocs, nothing, new_tester_2, 4.0),
        (false, :none, nothing, new_tester_3, Ref{Any}(StructFoo)),
        (false, :none, nothing, splatnew_tester, Ref{Tuple}((5.0, :a))),
        (false, :allocs, nothing, type_stable_getfield_tester_1, StableFoo(5.0, :hi)),
        (false, :allocs, nothing, type_stable_getfield_tester_2, StableFoo(5.0, :hi)),
        (false, :none, nothing, globalref_tester),
        (false, :none, nothing, globalref_tester_bool),
        (false, :none, nothing, globalref_tester_2, true),
        (false, :none, nothing, globalref_tester_2, false),
        (false, :allocs, nothing, globalref_tester_3),
        (false, :allocs, nothing, globalref_tester_4),
        (false, :none, (lb=1, ub=1_000), type_unstable_tester_0, Ref{Any}(5.0)),
        (false, :none, nothing, type_unstable_tester, Ref{Any}(5.0)),
        (false, :none, nothing, type_unstable_tester_2, Ref{Real}(5.0)),
        (false, :none, (lb=1, ub=500), type_unstable_tester_3, Ref{Any}(5.0)),
        (false, :none, (lb=1, ub=500), test_primitive_dynamic_dispatch, Any[5.0, false]),
        (false, :none, nothing, type_unstable_function_eval, Ref{Any}(sin), 5.0),
        (false, :allocs, nothing, phi_const_bool_tester, 5.0),
        (false, :allocs, nothing, phi_const_bool_tester, -5.0),
        (false, :allocs, nothing, phi_node_with_undefined_value, true, 4.0),
        (false, :allocs, nothing, phi_node_with_undefined_value, false, 4.0),
        (false, :allocs, nothing, test_multiple_phinode_block, 3.0, 3),
        (
            false,
            :none,
            nothing,
            Base._unsafe_getindex,
            IndexLinear(),
            randn(5),
            1,
            Base.Slice(Base.OneTo(1)),
        ), # fun PhiNode example
        (false, :allocs, nothing, avoid_throwing_path_tester, 5.0),
        (false, :allocs, nothing, simple_foreigncall_tester, randn(5)),
        (false, :none, nothing, simple_foreigncall_tester_2, randn(6), (2, 3)),
        (false, :allocs, nothing, foreigncall_tester, randn(5)),
        (false, :none, nothing, no_primitive_inlining_tester, 5.0),
        (false, :allocs, nothing, varargs_tester, 5.0),
        (false, :allocs, nothing, varargs_tester, 5.0, 4),
        (false, :allocs, nothing, varargs_tester, 5.0, 4, 3.0),
        (false, :allocs, nothing, varargs_tester_2, 5.0),
        (false, :allocs, nothing, varargs_tester_2, 5.0, 4),
        (false, :allocs, nothing, varargs_tester_2, 5.0, 4, 3.0),
        (false, :allocs, nothing, varargs_tester_3, 5.0),
        (false, :allocs, nothing, varargs_tester_3, 5.0, 4),
        (false, :allocs, nothing, varargs_tester_3, 5.0, 4, 3.0),
        (false, :allocs, nothing, varargs_tester_4, 5.0),
        (false, :allocs, nothing, varargs_tester_4, 5.0, 4),
        (false, :allocs, nothing, varargs_tester_4, 5.0, 4, 3.0),
        (false, :allocs, nothing, splatting_tester, 5.0),
        (false, :allocs, nothing, splatting_tester, (5.0, 4.0)),
        (false, :allocs, nothing, splatting_tester, (5.0, 4.0, 3.0)),
        # (false, :stability, nothing, unstable_splatting_tester, Ref{Any}(5.0)), # known failure case -- no rrule for _apply_iterate
        # (false, :stability, nothing, unstable_splatting_tester, Ref{Any}((5.0, 4.0))), # known failure case -- no rrule for _apply_iterate
        # (false, :stability, nothing, unstable_splatting_tester, Ref{Any}((5.0, 4.0, 3.0))), # known failure case -- no rrule for _apply_iterate
        (false, :none, (lb=1, ub=1_000), inferred_const_tester, Ref{Any}(nothing)),
        (false, :none, (lb=1, ub=1_000), datatype_slot_tester, 1),
        (false, :none, (lb=1, ub=1_000), datatype_slot_tester, 2),
        (false, :none, (lb=1, ub=100_000_000), test_union_of_arrays, randn(5), true),
        (
            false, :none, nothing,
            test_union_of_types, Ref{Union{Type{Float64}, Type{Int}}}(Float64),
        ),
        (false, :allocs, nothing, test_self_reference, 1.1, 1.5),
        (false, :allocs, nothing, test_self_reference, 1.5, 1.1),
        (false, :none, nothing, test_recursive_sum, randn(2)),
        (
            false, :none, nothing,
            LinearAlgebra._modify!,
            LinearAlgebra.MulAddMul(5.0, 4.0),
            5.0,
            randn(5, 4),
            (5, 4),
        ), # for Bool comma,
        (false, :allocs, nothing, getfield_tester, (5.0, 5)),
        (false, :allocs, nothing, getfield_tester_2, (5.0, 5)),
        (
            false, :allocs, nothing,
            setfield_tester_left!, FullyInitMutableStruct(5.0, randn(3)), 4.0,
        ),
        (
            false, :none, nothing,
            setfield_tester_right!, FullyInitMutableStruct(5.0, randn(3)), randn(5),
        ),
        (false, :none, nothing, mul!, randn(3, 5)', randn(5, 5), randn(5, 3), 4.0, 3.0),
        (false, :none, nothing, Random.make_seed, 5),
        (false, :none, nothing, Random.SHA.digest!, Random.SHA.SHA2_256_CTX()),
        (false, :none, nothing, Xoshiro, 123456),
        (false, :none, nothing, *, randn(250, 500), randn(500, 250)),
        (false, :allocs, nothing, test_sin, 1.0),
        (false, :allocs, nothing, test_cos_sin, 2.0),
        (false, :allocs, nothing, test_isbits_multiple_usage, 5.0),
        (false, :allocs, nothing, test_isbits_multiple_usage_2, 5.0),
        (false, :allocs, nothing, test_isbits_multiple_usage_3, 4.1),
        (false, :allocs, nothing, test_isbits_multiple_usage_4, 5.0),
        (false, :allocs, nothing, test_isbits_multiple_usage_5, 4.1),
        (false, :allocs, nothing, test_isbits_multiple_usage_phi, false, 1.1),
        (false, :allocs, nothing, test_isbits_multiple_usage_phi, true, 1.1),
        (false, :allocs, nothing, test_multiple_call_non_primitive, 5.0),
        (false, :none, (lb=1, ub=1500), test_multiple_pi_nodes, Ref{Any}(5.0)),
        (false, :none, (lb=1, ub=500), test_multi_use_pi_node, Ref{Any}(5.0)),
        (false, :allocs, nothing, test_getindex, [1.0, 2.0]),
        (false, :allocs, nothing, test_mutation!, [1.0, 2.0]),
        (false, :allocs, nothing, test_while_loop, 2.0),
        (false, :allocs, nothing, test_for_loop, 3.0),
        (false, :none, nothing, test_mutable_struct_basic, 5.0),
        (false, :none, nothing, test_mutable_struct_basic_sin, 5.0),
        (false, :none, nothing, test_mutable_struct_setfield, 4.0),
        (false, :none, (lb=1, ub=500), test_mutable_struct, 5.0),
        (false, :none, nothing, test_struct_partial_init, 3.5),
        (false, :none, nothing, test_mutable_partial_init, 3.3),
        (
            false, :allocs, nothing,
            test_naive_mat_mul!, randn(100, 50), randn(100, 30), randn(30, 50),
        ),
        (
            false, :allocs, nothing,
            (A, C) -> test_naive_mat_mul!(C, A, A), randn(25, 25), randn(25, 25),
        ),
        (false, :allocs, nothing, sum, randn(32)),
        (false, :none, nothing, test_diagonal_to_matrix, Diagonal(randn(30))),
        (
            false, :allocs, nothing,
            ldiv!, randn(20, 20), Diagonal(rand(20) .+ 1), randn(20, 20),
        ),
        (
            false, :allocs, nothing,
            LinearAlgebra._kron!, randn(25, 25), randn(5, 5), randn(5, 5),
        ),
        (
            false, :allocs, nothing,
            kron!, randn(25, 25), Diagonal(randn(5)), randn(5, 5),
        ),
        (
            false, :none, nothing,
            test_mlp,
            randn(sr(1), 50, 20),
            randn(sr(2), 70, 50),
            randn(sr(3), 30, 70),
        ),
        (false, :allocs, nothing, test_handwritten_sum, randn(128, 128)),
        (false, :allocs, nothing, _naive_map_sin_cos_exp, randn(1024), randn(1024)),
        (false, :allocs, nothing, _naive_map_negate, randn(1024), randn(1024)),
        (false, :allocs, nothing, test_from_slack, randn(10_000)),
        (false, :none, nothing, _sum, randn(1024)),
        (false, :none, nothing, test_map, randn(1024), randn(1024)),
        (false, :none, nothing, _broadcast_sin_cos_exp, randn(10, 10)),
        (false, :none, nothing, _map_sin_cos_exp, randn(10, 10)),
        (false, :none, nothing, ArgumentError, "hi"),
        (false, :none, nothing, test_small_union, Ref{Union{Float64, Vector{Float64}}}(5.0)),
        (false, :none, nothing, test_small_union, Ref{Union{Float64, Vector{Float64}}}([1.0])),
        (false, :allocs, nothing, inlinable_invoke_call, 5.0),
        (false, :none, nothing, inlinable_vararg_invoke_call, (2, 2), 5.0, 4.0, 3.0, 2.0),
        (false, :none, nothing, hvcat, (2, 2), 3.0, 2.0, 0.0, 1.0),
    ]
end

_broadcast_sin_cos_exp(x::AbstractArray{<:Real}) = sum(sin.(cos.(exp.(x))))

_map_sin_cos_exp(x::AbstractArray{<:Real}) = sum(map(x -> sin(cos(exp(x))), x))

function _naive_map_sin_cos_exp(y::AbstractArray{<:Real}, x::AbstractArray{<:Real})
    n = 1
    while n <= length(x)
        y[n] = sin(cos(exp(x[n])))
        n += 1
    end
    return y
end

function _naive_map_negate(y::AbstractArray{<:Real}, x::AbstractArray{<:Real})
    n = 1
    while n <= length(x)
        y[n] = -x[n]
        n += 1
    end
    return y
end

function test_from_slack(x::AbstractVector{T}) where {T}
    y = zero(T)
    n = 1
    while n <= length(x)
        if iseven(n)
            y += sin(x[n])
        else
            y += cos(x[n])
        end
        n += 1
    end
    return y
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

function Tapir.rrule!!(::Tapir.CoDual{typeof(my_setfield!)}, value, name, x)
    _name = primal(name)
    old_x = isdefined(primal(value), _name) ? getfield(primal(value), _name) : nothing
    function setfield!_pullback(dy, df, dvalue, ::NoTangent, dx)
        new_dx = increment!!(dx, val(getfield(dvalue.fields, _name)))
        new_dx = increment!!(new_dx, dy)
        old_x !== nothing && setfield!(primal(value), _name, old_x)
        return df, dvalue, NoTangent(), new_dx
    end
    y = Tapir.CoDual(
        setfield!(primal(value), _name, primal(x)),
        _setfield!(tangent(value), _name, tangent(x)),
    )
    return y, setfield!_pullback
end

# Tests brought in from DiffTests.jl
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

export MutableFoo, StructFoo, NonDifferentiableFoo, FullyInitMutableStruct

end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:test_utils})
    return TestResources.generate_test_functions(), Any[]
end

using .TestResources
