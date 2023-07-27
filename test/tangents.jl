function test_increment!!(z_target::T, x::T, y::T) where {T}
    function check_aliasing(z::T, x::T) where {T}
        if ismutabletype(T)
            return z === x
        else
            tmp = map(fieldnames(T)) do f
                return check_aliasing(getfield(z, f), getfield(x, f))
            end
            return all(tmp)
        end
    end

    z = Taped.increment!!(x, y)
    @test check_aliasing(z, x)
    @test z == z_target
end

tangent(nt::NamedTuple) = Tangent(map(PossiblyUninitTangent, nt))
mutable_tangent(nt::NamedTuple) = MutableTangent(map(PossiblyUninitTangent, nt))

function test_tangent(rng::AbstractRNG, p::P, z_target::T, x::T, y::T) where {P, T}

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
    @test z == Taped.zero_tangent(p)

    # Verify that the zero tangent is zero via its action.
    zc = deepcopy(z)
    tc = deepcopy(t)
    @test increment!!(zc, zc) == zc
    @test increment!!(zc, tc) == tc
    @test increment!!(tc, zc) == tc

    if ismutabletype(P)
        @test increment!!(zc, zc) === zc
        @test increment!!(tc, zc) === tc
        @test increment!!(zc, tc) === zc
        @test increment!!(tc, tc) === tc
    end

    z_pred = increment!!(x, y)
    @test z_pred == z_target
    if ismutabletype(P)
        @test z_pred === x
    end

    # If t isn't the zero element, then adding it to itself must change its value.
    if t != z
        if !ismutabletype(P)
            @test !(increment!!(tc, tc) == tc)
        end
    end

    # Adding things preserves types.
    @test increment!!(zc, zc) isa Tt
    @test increment!!(zc, tc) isa Tt
    @test increment!!(tc, zc) isa Tt

    # Setting to zero equals zero.
    @test set_to_zero!!(tc) == z
    if ismutabletype(P)
        @test set_to_zero!!(tc) === tc
    end
end

function test_test_interface(p::P, t::T) where {P, T}
    @assert tangent_type(P) == T
    @test _scale(2.0, t) isa T
    @test _dot(t, t) isa Float64
    @test _dot(t, t) >= 0.0
    @test _dot(t, zero_tangent(p)) == 0.0
    @test _dot(t, increment!!(deepcopy(t), t)) ≈ 2 * _dot(t, t)
    @test _add_to_primal(p, t) isa P
    @test _add_to_primal(p, zero_tangent(p)) == p
    @test _diff(p, p) isa T
    @test _diff(p, p) == zero_tangent(p)
end

@testset "tangents" begin
    rng = Xoshiro(123456)
    @testset "sin" begin
        t = Tangent((;))
        test_tangent(rng, sin, t, t, t)
        test_test_interface(sin, t)
    end

    # NOTE: ADD INTERFACE-ONLY TEST FOR LAZY STRING

    @testset "$T" for T in [Float16, Float32, Float64]
        test_tangent(rng, T(10), T(9), T(5), T(4))
        test_test_interface(T(10), T(9))
    end

    @testset "Vector{Float64}" begin
        p = randn(5)
        x = randn(5)
        y = randn(5)
        test_tangent(rng, p, x + y, x, y)
        test_test_interface(p, x)
    end

    @testset "Vector{Int}" begin
        p = rand(Int, 5)
        x = fill(NoTangent(), 5)
        y = fill(NoTangent(), 5)
        z = fill(NoTangent(), 5)
        # x = rand(Int, 5)
        # y = rand(Int, 5)
        test_tangent(rng, p, z, x, y)
        test_test_interface(p, x)
    end

    @testset "Tuple{Float64, Vector{Float64}}" begin
        p = (6.0, randn(5))
        x = (5.0, randn(5))
        y = (4.0, randn(5))
        z = (9.0, x[2] + y[2])
        test_tangent(rng, p, z, x, y)
        test_test_interface(p, x)
    end

    @testset "NamedTuple{(:a, :b), Tuple{Float64, Vector{Float64}}}" begin
        p = (a=6.0, b=randn(5))
        x = (a=5.0, b=randn(5))
        y = (a=4.0, b=rand(5))
        z = (a=9.0, b=x.b + y.b)
        test_tangent(rng, p, z, x, y)
        test_test_interface(p, x)
    end

    @testset "NamedTuple{(), Tuple{}}" begin
        p = (;)
        x = (;)
        test_tangent(rng, p, x, x, x)
    end

    @testset "StructFoo (full init)" begin
        p = TestResources.StructFoo(6.0, randn(5))
        _tangent = nt -> Taped.build_tangent(typeof(p), nt...)
        x = _tangent((a=5.0, b=randn(5)))
        y = _tangent((a=4.0, b=randn(5)))
        z = _tangent((a=9.0, b=x.fields.b.tangent + y.fields.b.tangent))
        test_tangent(rng, p, z, x, y)
        test_test_interface(p, x)

        # Verify tangent generation works correctly.
        t = _tangent((a=5.0, b=randn(5)))
        @test Taped.is_init(t.fields.a) == true
        @test Taped.is_init(t.fields.b) == true
    end

    T_b = PossiblyUninitTangent{Vector{Float64}}
    @testset "StructFoo (partial init)" begin
        p = TestResources.StructFoo(6.0)
        _tangent = nt -> Taped.build_tangent(typeof(p), nt...)
        x = _tangent((a=5.0,))
        y = _tangent((a=4.0,))
        z = _tangent((a=9.0,))
        test_tangent(rng, p, z, x, y)

        x_init = _tangent((a=5.0, b=randn(5)))
        @test_throws ErrorException increment!!(x_init, x)
        @test_throws ErrorException increment!!(x, x_init)

        # Verify tangent generation works correctly.
        t = Taped.build_tangent(TestResources.StructFoo, 5.0)
        @test Taped.is_init(t.fields.a) == true
        @test Taped.is_init(t.fields.b) == false
    end

    @testset "MutableFoo (full init)" begin
        p = TestResources.MutableFoo(6.0, randn(5))
        x = mutable_tangent((a=5.0, b=randn(5)))
        y = mutable_tangent((a=4.0, b=rand(5)))
        z = mutable_tangent((a=9.0, b=x.fields.b.tangent + y.fields.b.tangent))
        test_tangent(rng, p, z, x, y)
        test_test_interface(p, x)

        # Verify tangent generation works correctly.
        t = Taped.build_tangent(TestResources.MutableFoo, 5.0, randn(5))
        @test Taped.is_init(t.fields.a) == true
        @test Taped.is_init(t.fields.b) == true
    end

    @testset "MutableFoo (partial init)" begin
        p = TestResources.MutableFoo(6.0)
        x = MutableTangent((a=_wrap_field(5.0), b=T_b()))
        y = MutableTangent((a=_wrap_field(4.0), b=T_b()))
        z = MutableTangent((a=_wrap_field(9.0), b=T_b()))
        test_tangent(rng, p, z, x, y)

        x_init = mutable_tangent((a=5.0, b=randn(5)))
        @test_throws ErrorException increment!!(x_init, x)
        @test_throws ErrorException increment!!(x, x_init)

        # Verify tangent generation works correctly.
        t = Taped.build_tangent(TestResources.MutableFoo, 5.0)
        @test Taped.is_init(t.fields.a) == true
        @test Taped.is_init(t.fields.b) == false
    end

    @testset "UnitRange{Int}" begin
        p = UnitRange{Int}(5, 7)
        x = Taped.build_tangent(typeof(p), NoTangent(), NoTangent())
        test_tangent(rng, p, x, x, x)

        # Verify tangent generation works correctly.
        t = Taped.build_tangent(UnitRange{Int}, NoTangent(), NoTangent())
        @test Taped.is_init(t.fields.start) == true
        @test Taped.is_init(t.fields.stop) == true
    end

    @testset "types" begin
        @testset "$T" for T in [Array, Float64, Union{Float64, Float32}, Union, UnionAll]
            test_tangent(rng, T, NoTangent(), NoTangent(), NoTangent())
        end
    end

    test_tangent(rng, Core.Intrinsics.xor_int, NoTangent(), NoTangent(), NoTangent())
    test_tangent(rng, typeof(<:), NoTangent(), NoTangent(), NoTangent())

    @testset "increment_field!!" begin
        @testset "NoTangent" begin
            @test increment_field!!(NoTangent(), NoTangent(), :a) == NoTangent()
        end
        @testset "Tuple" begin
            x = (5.0, 4.0)
            y = 3.0
            @test increment_field!!(x, y, 1) == (8.0, 4.0)
            @test increment_field!!(x, y, 2) == (5.0, 7.0)
        end
        @testset "NamedTuple" begin
            x = (a=5.0, b=4.0)
            y = 3.0
            @test increment_field!!(x, y, :a) == (a=8.0, b=4.0)
            @test increment_field!!(x, y, :b) == (a=5.0, b=7.0)
        end
        @testset "Tangent" begin
            x = tangent((a=5.0, b=4.0))
            y = 3.0
            @test increment_field!!(x, y, :a) == tangent((a=8.0, b=4.0))
            @test increment_field!!(x, y, :b) == tangent((a=5.0, b=7.0))
        end
        @testset "MutableTangent" begin
            x = mutable_tangent((a=5.0, b=4.0))
            y = 3.0
            @test increment_field!!(x, y, :a) == mutable_tangent((a=8.0, b=4.0))
            @test increment_field!!(x, y, :a) === x

            x = mutable_tangent((a=5.0, b=4.0))
            y = 3.0
            @test increment_field!!(x, y, :b) == mutable_tangent((a=5.0, b=7.0))
            @test increment_field!!(x, y, :b) === x
        end
    end

    @testset "set_field_to_zero!!" begin
        nt = (a=5.0, b=4.0)
        nt2 = (a=0.0, b=4.0)
        @test set_field_to_zero!!(nt, :a) == nt2
        @test set_field_to_zero!!((5.0, 4.0), 2) == (5.0, 0.0)
        @test set_field_to_zero!!(tangent(nt), :a) == tangent(nt2)

        x = mutable_tangent(nt)
        @test set_field_to_zero!!(x, :a) == mutable_tangent(nt2)
        @test set_field_to_zero!!(x, :a) === x
    end
end


# The goal of these tests is to check that we can indeed generate tangent types for anything
# that we will encounter in the Julia language. We try to achieve this by pulling in types
# from LinearAlgebra, and Random, and generating tangents for them.
# Need to improve coverage of Base / Core really.
# It was these tests which made me aware of the issues around uninitialised values.

#
# Utility functionality for getting good test coverage of the types in a given module.
#

# function items_in(m::Module)
#     if m === Core.Compiler
#         @info "not getting names in Core"
#         return []
#     end
#     @info "gathering items in $m"
#     defined_names = filter(name -> isdefined(m, name), names(m; all=true))
#     return map(name -> getfield(m, name), defined_names)
# end

# function modules_in(m::Module)
#     if m === Core
#         @info "not getting names in Core"
#         return []
#     end
#     @info "gathering items in $m"
#     defined_names = filter(name -> isdefined(m, name), names(m; all=false))
#     items = map(name -> getfield(m, name), defined_names)
#     return filter(item -> item isa Module && item !== m, items)
# end

# function types_in(m::Module)
#     return filter(item -> item isa Type, items_in(m))
# end

# function concrete_types_in(m::Module)
#     return filter(isconcretetype, types_in(m))
# end

# function struct_types_in(m::Module)
#     return filter(isstructtype, types_in(m))
# end

# function primitive_types_in(m::Module)
#     return filter(isprimitivetype, types_in(m))
# end

# function non_abstract_types_in(m::Module)
#     return filter(!isabstracttype, types_in(m))
# end


# # Primitives are required to explicitly declare a method of `zero_tangent` which applies
# # to them. They must not hit the generic fallback. This function checks that there are no
# # primitives within the specified module which don't hit a generic fallback.
# function verify_zero_for_primitives_in(m::Module)
#     return filter(t -> isprimitivetype(t), types_in(m))
# end


# # A toy type on which to test tangent stuff in a variety of situations.
# struct Foo{T, V}
#     x::T
#     y::V
#     Foo{T, V}(x, y) where {T, V} = new{T, V}(x, y)
#     Foo{T, V}(x) where {T, V} = new{T, V}(x)
#     Foo(x, y) = new{typeof(x), typeof(y)}(x, y)
# end

# function test_basic_tangents(rng::AbstractRNG)
#     @testset "typeof($x)" for x in [
#         Foo(5.0, 5.0),
#         Foo{Float64, Ref{Float64}}(5.0),
#         Foo{Any, Ref{Float64}}(5.0),
#     ]
#         test_tangent(rng, x, 1e-3)
#     end
# end

# function test_tricky_tangents(rng::AbstractRNG)
#     @testset "typeof($x)" for x in [
#         Foo(5.0, 5.0),
#         Foo{Float64, Int64}(5.0),
#         Foo{Int64, Int64}(5, 4),
#         Foo{Real, Float64}(5.0, 4.0),
#         Foo{Real, Float64}(5.0),
#         Foo{Float64, Float64}(5.0),
#         Foo{Float64, Vector{Float64}}(5.0),
#         Foo{Real, Real}(5.0),
#         Foo{Union{Float64, Int64}, Int64}(5.0, 4),
#         Foo{Union{Float64, Int64}, Int64}(5.0),
#         Foo{DataType, Float64}(Float64),
#         Foo{DataType, Int64}(Float64),
#     ]
#         test_tangent(rng, x, 1e-3)
#     end
# end

# function primitives_for_LinearAlgebra()
#     return [
#         Adjoint(randn(4, 5)),
#         Bidiagonal(randn(5, 5), :U),
#         bunchkaufman(Symmetric(randn(5, 5))),
#         cholesky(collect(Matrix{Float64}(I, 3, 3))),
#         cholesky(collect(Matrix{Float64}(I, 3, 3)), Val(:true)),
#         ColumnNorm(),
#         Diagonal(randn(4)),
#         eigen(randn(3, 3)),
#         GeneralizedEigen(randn(3), randn(3, 3)),
#         svd(randn(3, 3), randn(3, 3)),
#         schur(randn(3, 3), randn(3, 3)),
#         Hermitian(randn(3, 3)),
#         hessenberg(randn(3, 3)),
#         LAPACKException(5),
#         ldlt(SymTridiagonal(randn(3), randn(2))),
#         lq(randn(3, 3)),
#         lu(randn(3, 3)),
#         LowerTriangular(randn(3, 3)),
#         NoPivot(),
#         PosDefException(5),
#         qr(randn(2, 4)),
#         qr(randn(2, 4), ColumnNorm()),
#         RankDeficientException(5),
#         RowMaximum(),
#         svd(randn(2, 3)),
#         schur(randn(2, 2)),
#         SingularException(5),
#         SymTridiagonal(randn(3), randn(2)),
#         Symmetric(randn(3, 3)),
#         Transpose(randn(3, 3)),
#         Tridiagonal(randn(5, 5)),
#         UniformScaling(5.0),
#         UnitLowerTriangular(randn(5, 5)),
#         UnitUpperTriangular(randn(5, 5)),
#         UpperHessenberg(randn(5, 5)),
#         UpperTriangular(randn(3, 3)),
#         ZeroPivotException(5),
#     ]
# end

# function primitives_for_Random()
#     return [
#         MersenneTwister(1), Xoshiro(1), RandomDevice(), TaskLocalRNG()
#     ]
# end

# function test_many_tangents(rng)
#     @testset "$(typeof(x))" for x in vcat(
#         [
#             Int8(5), Int16(5), Int32(5), Int64(5), Int128(5),
#             UInt8(5), UInt16(5), UInt32(5), UInt64(5), UInt128(5),
#             Float16(5.0), Float32(4.0), Float64(3.0), BigFloat(10.0),
#         ],
#         [randn(T, inds...)
#             for T in [Float16, Float32, Float64]
#             for inds in ((3, ), (3, 4), (3, 4, 5))
#         ],
#         [
#             fill(1, 3), fill(1, 3, 4), fill(1, 4, 5, 6),
#         ],
#         primitives_for_LinearAlgebra(),
#         primitives_for_Random(),
#     )
#         test_tangent(rng, x, 1e-3)
#     end
# end

# # Verify that the tangent type of everything in the `m` can be determined.
# # This is a surprisingly hard condition to satisfy.
# function get_unique_tangent_types(m::Module)
#     types_in_m = filter(x -> x isa Type, items_in(m))
#     unique(map(tangent_type, types_in_m))
#     return unique(map(tangent_type ∘ typeof, items_in(m)))
# end

# # Recursively gather all of the objects in all of the modules accessible from `m`.
# # Most of these things will be types, functions, etc, rather than instances of them.
# # This is useful, because it's lots of random stuff that you don't typically think of as
# # being AD-related, and should mostly have `NoTangent` tangents.
# # Test this to ensure that that actually happens!
# function gather_objects(m::Module)
#     get_unique_tangent_types(m)
#     foreach(gather_objects, modules_in(m))
# end

# @testset "tangent_types" begin
#     test_basic_tangents(Xoshiro(123456))
#     test_many_tangents(Xoshiro(123456))
#     test_tricky_tangents(Xoshiro(123456))
# end
