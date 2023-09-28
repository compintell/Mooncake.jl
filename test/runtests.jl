using
    BenchmarkTools,
    FunctionWrappers,
    LinearAlgebra,
    Random,
    Taped,
    Test,
    Umlaut

using Base: unsafe_load, pointer_from_objref
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
    set_shadow!!

using .TestUtils:
    test_rrule!!,
    test_taped_rrule!!,
    has_equal_data,
    AddressMap,
    populate_address_map!,
    populate_address_map

@testset "Taped.jl" begin
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
        @info "misc"
        include(joinpath("rrules", "misc.jl"))
        @info "umlaut_internals_rules"
        include(joinpath("rrules", "umlaut_internals_rules.jl"))
        @info "battery_tests"
        include(joinpath("rrules", "battery_tests.jl"))
        @info "unrolled_function"
        include(joinpath("rrules", "unrolled_function.jl"))
    end
end
