using
    BenchmarkTools,
    DiffRules,
    FillArrays,
    JET,
    LinearAlgebra,
    PDMats,
    Random,
    SpecialFunctions,
    StableRNGs,
    Taped,
    Test

using Base: unsafe_load, pointer_from_objref
using Base.Iterators: product
using Core:
    bitcast, svec, ReturnNode, PhiNode, PiNode, GotoIfNot, GotoNode, SSAValue, Argument
using Core.Intrinsics: pointerref, pointerset

using Taped:
    CC,
    IntrinsicsWrappers,
    TestUtils,
    TestResources,
    CoDual,
    _wrap_field,
    DefaultCtx,
    rrule!!,
    lgetfield,
    lsetfield!,
    might_be_active,
    build_tangent,
    SlotRef,
    ConstSlot,
    TypedGlobalRef,
    build_inst,
    TypedPhiNode,
    build_coinsts,
    Stack,
    _typeof

using .TestUtils:
    test_rrule!!,
    has_equal_data,
    AddressMap,
    populate_address_map!,
    populate_address_map,
    test_tangent

using .TestResources:
    TypeStableMutableStruct,
    StructFoo,
    MutableFoo

# The integration tests take ages to run, so we split them up. CI sets up two jobs -- the
# "basic" group runs test that, when passed, _ought_ to imply correctness of the entire
# scheme. The "extended" group runs a large battery of tests that should pick up on anything
# that has been missed in the "basic" group. As a rule, if the "basic" group passes, but the
# "extended" group fails, there are clearly new tests that need to be added to the "basic"
# group.
const test_group = get(ENV, "TEST_GROUP", "basic")

sr(n::Int) = StableRNG(n)

# This is annoying and hacky and should be improved.
if isempty(Taped.TestTypes.PRIMALS)
    Taped.TestTypes.generate_primals()
end

@testset "Taped.jl" begin
    if test_group == "basic"
        include("utils.jl")
        include("tangents.jl")
        include("codual.jl")
        include("stack.jl")
        @testset "interpreter" begin
            include(joinpath("interpreter", "contexts.jl"))
            include(joinpath("interpreter", "ir_utils.jl"))
            include(joinpath("interpreter", "ir_normalisation.jl"))
            include(joinpath("interpreter", "abstract_interpretation.jl"))
            include(joinpath("interpreter", "interpreted_function.jl"))
            include(joinpath("interpreter", "reverse_mode_ad.jl"))
        end
    elseif test_group == "rrules"
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
            @info "new"
            include(joinpath("rrules", "new.jl"))
        end
    elseif test_group == "integration_testing/misc"
        include(joinpath("integration_testing/", "misc.jl"))
        include(joinpath("integration_testing", "battery_tests.jl"))
    elseif test_group == "integration_testing/diff_tests"
        include(joinpath("integration_testing", "diff_tests.jl"))
    elseif test_group == "integration_testing/distributions"
        include(joinpath("integration_testing", "distributions.jl"))
    elseif test_group == "integration_testing/gp"
        include(joinpath("integration_testing", "gp.jl"))
    elseif test_group == "integration_testing/special_functions"
        include(joinpath("integration_testing", "special_functions.jl"))
    elseif test_group == "integration_testing/array"
        include(joinpath("integration_testing", "array.jl"))
    elseif test_group == "integration_testing/turing"
        include(joinpath("integration_testing", "turing.jl"))
    else
        throw(error("test_group=$(test_group) is not recognised"))
    end
end
