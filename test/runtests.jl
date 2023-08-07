using
    LinearAlgebra,
    Random,
    Taped,
    Test,
    Umlaut

using Base: unsafe_load, pointer_from_objref
using Core: bitcast
using Core: Intrinsics
using Core.Intrinsics: pointerref
using Taped: TestUtils, CoDual, to_reverse_mode_ad, _wrap_field, __intrinsic__
using .TestUtils: test_rrule!!, test_taped_rrule!!

include("test_resources.jl")

@testset "Taped.jl" begin
    include("tracing.jl")
    include("tangents.jl")
    include("reverse_mode_ad.jl")
    @testset "rrules" begin
        include(joinpath("rrules", "avoiding_non_differentiable_code.jl"))
        include(joinpath("rrules", "blas.jl"))
        include(joinpath("rrules", "builtins.jl"))
        include(joinpath("rrules", "foreigncall.jl"))
        include(joinpath("rrules", "misc.jl"))
        include(joinpath("rrules", "umlaut_internals_rules.jl"))
        # include(joinpath("rrules", "unrolled_function.jl"))
    end
end
