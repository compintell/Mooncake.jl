using
    BenchmarkTools,
    DiffRules,
    FunctionWrappers,
    JET,
    LinearAlgebra,
    Random,
    Taped,
    Test,
    Umlaut

using Base: unsafe_load, pointer_from_objref
using Base.Iterators: product
using Core: bitcast
using Core.Intrinsics: pointerref, pointerset
using FunctionWrappers: FunctionWrapper

using Taped:
    IntrinsicsWrappers,
    TestUtils,
    TestResources,
    CoDual,
    to_reverse_mode_ad,
    _wrap_field,
    build_coinstruction,
    const_coinstruction,
    input_primals,
    input_shadows,
    output_primal,
    output_shadow,
    pullback!,
    seed_output_shadow!,
    rrule!!,
    set_shadow!!,
    SSym,
    SInt,
    lgetfield,
    might_be_active,
    rebind

using .TestUtils:
    test_rrule!!,
    test_taped_rrule!!,
    has_equal_data,
    AddressMap,
    populate_address_map!,
    populate_address_map,
    test_tangent,
    test_numerical_testing_interface

# The integration tests take ages to run, so we split them up. CI sets up two jobs -- the
# "basic" group runs test that, when passed, _ought_ to imply correctness of the entire
# scheme. The "extended" group runs a large battery of tests that should pick up on anything
# that has been missed in the "basic" group. As a rule, if the "basic" group passes, but the
# "extended" group fails, there are clearly new tests that need to be added to the "basic"
# group.
const test_group = get(ENV, "TEST_GROUP", "basic")

@testset "Taped.jl" begin
    if test_group == "basic"
        include("tracing.jl")
        include("acceleration.jl")
        include("tangents.jl")
        include("reverse_mode_ad.jl")
        include("test_utils.jl")
        @testset "rrules" begin
            @info "avoiding_non_differentiable_code"
            include(joinpath("rrules", "avoiding_non_differentiable_code.jl"))
            @info "blas"
            include(joinpath("rrules", "blas.jl"))
            @info "builtins"
            include(joinpath("rrules", "builtins.jl"))
            @info "foreigncall"
            include(joinpath("rrules", "foreigncall.jl"))
            @info "lapack"
            include(joinpath("rrules", "lapack.jl"))
            @info "low_level_maths"
            include(joinpath("rrules", "low_level_maths.jl"))
            @info "misc"
            include(joinpath("rrules", "misc.jl"))
            @info "umlaut_internals_rules"
            include(joinpath("rrules", "umlaut_internals_rules.jl"))
            @info "battery_tests"
            include(joinpath("rrules", "battery_tests.jl"))
            @info "unrolled_function"
            include(joinpath("rrules", "unrolled_function.jl"))
        end
    elseif test_group == "extended"
        include(joinpath("rrules", "integration_testing.jl"))
    else
        throw(error("test_group=$(test_group) is not recognised"))
    end
end