using
    LinearAlgebra,
    Random,
    Taped,
    Test,
    Umlaut

using Base: unsafe_load, pointer_from_objref
using Core: bitcast
using Core: Intrinsics
using Core.Intrinsics: pointerref, pointerset
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
        include(joinpath("rrules", "unrolled_function.jl"))
    end
end


# v_fargs = unsplat!(t, v_fargs)
# # note: we need to extract IR before vararg grouping, which may change
# # v_fargs, thus invalidating method search
# ir = getcode(code_signature(t.tape.c, v_fargs)...)
# sparams, sparams_dict = get_static_params(t, v_fargs)
# v_fargs = group_varargs!(t, v_fargs)
# frame = Frame(t.tape, ir, v_fargs...)
# push!(t.stack, frame)


# -macro __new__(T, arg)
# -    esc(Expr(:new, T, arg))
# end

# -# """
# -#     __new__(T, args...)
# -# User-level version of the `new()` pseudofunction.
# -# Can be used to construct most Julia types, including structs
# -# without default constructors, closures, etc.
# -# """
# -# @inline function __new__(T, args...)
# -#     @__splatnew__(T, args)
# -# end
# -
# -@inline @generated __new__(T, x...) = Expr(:new, :T, map(n -> :(x[$n]), 1:length(x))...)
# -# @inline @generated __splatnew__(T, x...) = Expr(:splatnew, :T, :x)
