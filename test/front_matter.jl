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
    _typeof,
    FwdStack

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
