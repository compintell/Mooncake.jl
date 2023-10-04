tangent(nt::NamedTuple) = Tangent(map(PossiblyUninitTangent, nt))
mutable_tangent(nt::NamedTuple) = MutableTangent(map(PossiblyUninitTangent, nt))
function mutable_tangent(::Type{P}, nt::NamedTuple{names}) where {P, names}
    v = map((P, x) -> PossiblyUninitTangent{tangent_type(P)}(x), fieldtypes(P), nt)
    return MutableTangent(NamedTuple{names}(v))
end

@testset "tangents" begin
    rng = Xoshiro(123456)
    @testset "sin" begin
        t = Tangent((;))
        test_tangent(rng, sin, t, t, t)
        test_numerical_testing_interface(sin, t)
    end

    # NOTE: ADD INTERFACE-ONLY TEST FOR LAZY STRING

    @testset "$T" for T in [Float16, Float32, Float64]
        test_tangent(rng, T(10), T(9), T(5), T(4))
        test_numerical_testing_interface(T(10), T(9))
    end

    @testset "Vector{Float64}" begin
        p = randn(5)
        x = randn(5)
        y = randn(5)
        test_tangent(rng, p, x + y, x, y)
        test_numerical_testing_interface(p, x)
    end

    @testset "Vector{Int}" begin
        p = rand(Int, 5)
        x = fill(NoTangent(), 5)
        y = fill(NoTangent(), 5)
        z = fill(NoTangent(), 5)
        test_tangent(rng, p, z, x, y)
        test_numerical_testing_interface(p, x)
    end

    @testset "Vector{Vector{Float64}}" begin
        p = [randn(3) for _ in 1:4]
        x = [randn(3) for _ in 1:4]
        y = [randn(3) for _ in 1:4]
        z = x .+ y
        test_tangent(rng, p, z, x, y)
        test_numerical_testing_interface(p, x)
    end

    @testset "Vector{Vector{Float64}} with undefs" begin
        p = Vector{Vector{Float64}}(undef, 4)
        p[1] = randn(3)
        p[2] = randn(4)
        x = Vector{Vector{Float64}}(undef, 4)
        x[1] = randn(3)
        x[2] = randn(4)
        y = Vector{Vector{Float64}}(undef, 4)
        y[1] = randn(3)
        y[2] = randn(4)
        z = Vector{Vector{Float64}}(undef, 4)
        z[1] = x[1] + y[1]
        z[2] = x[2] + y[2]
        test_tangent(rng, p, z, x, y)
        test_numerical_testing_interface(p, x)
    end

    @testset "Tuple{Float64, Vector{Float64}}" begin
        p = (6.0, randn(5))
        x = (5.0, randn(5))
        y = (4.0, randn(5))
        z = (9.0, x[2] + y[2])
        test_tangent(rng, p, z, x, y)
        test_numerical_testing_interface(p, x)
    end

    @testset "NamedTuple{(:a, :b), Tuple{Float64, Vector{Float64}}}" begin
        p = (a=6.0, b=randn(5))
        x = (a=5.0, b=randn(5))
        y = (a=4.0, b=rand(5))
        z = (a=9.0, b=x.b + y.b)
        test_tangent(rng, p, z, x, y)
        test_numerical_testing_interface(p, x)
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
        test_numerical_testing_interface(p, x)

        # Verify tangent generation works correctly.
        t = _tangent((a=5.0, b=randn(5)))
        @test Taped.is_init(t.fields.a) == true
        @test Taped.is_init(t.fields.b) == true
    end

    T_b = PossiblyUninitTangent{Any}
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
        _tangent = (args...) -> Taped.build_tangent(TestResources.MutableFoo, args...)
        p = TestResources.MutableFoo(6.0, randn(5))
        x = _tangent(5.0, randn(5))
        y = _tangent(4.0, rand(5))
        z = _tangent(9.0, x.fields.b.tangent + y.fields.b.tangent)
        test_tangent(rng, p, z, x, y)
        test_numerical_testing_interface(p, x)

        # Verify tangent generation works correctly.
        t = _tangent(5.0, randn(5))
        @test Taped.is_init(t.fields.a) == true
        @test Taped.is_init(t.fields.b) == true
    end

    @testset "MutableFoo (partial init)" begin
        p = TestResources.MutableFoo(6.0)
        x = MutableTangent((a=_wrap_field(5.0), b=T_b()))
        y = MutableTangent((a=_wrap_field(4.0), b=T_b()))
        z = MutableTangent((a=_wrap_field(9.0), b=T_b()))
        test_tangent(rng, p, z, x, y)

        x_init = mutable_tangent(TestResources.MutableFoo, (a=5.0, b=randn(5)))
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

    @testset "zero_tangent performance" begin
        Taped.zero_tangent(Main)
        @test @allocated(Taped.zero_tangent(Main)) == 0
    end

    @testset "increment_field!!" begin
        @testset "NoTangent" begin
            nt = NoTangent()
            @test @inferred(increment_field!!(nt, nt, SSym(:a))) == nt
            @test increment_field!!(nt, nt, :a) == nt
            @test @inferred(increment_field!!(nt, nt, SInt(1))) == nt
            @test increment_field!!(nt, nt, 1) == nt
        end
        @testset "Tuple" begin
            nt = NoTangent()
            x = (5.0, nt)
            y = 3.0
            @test @inferred(increment_field!!(x, y, SInt(1))) == (8.0, nt)
            @test @inferred(increment_field!!(x, nt, SInt(2))) == (5.0, nt)

            # Slow versions.
            @test increment_field!!(x, y, 1) == (8.0, nt)
            @test increment_field!!(x, nt, 2) == (5.0, nt)
        end
        @testset "NamedTuple" begin
            nt = NoTangent()
            x = (a=5.0, b=nt)
            @test @inferred(increment_field!!(x, 3.0, SSym(:a))) == (a=8.0, b=nt)
            @test @inferred(increment_field!!(x, nt, SSym(:b))) == (a=5.0, b=nt)
            @test @inferred(increment_field!!(x, 3.0, SInt(1))) == (a=8.0, b=nt)
            @test @inferred(increment_field!!(x, nt, SInt(2))) == (a=5.0, b=nt)

            # Slow versions.
            @test increment_field!!(x, 3.0, :a) == (a=8.0, b=nt)
            @test increment_field!!(x, nt, :b) == (a=5.0, b=nt)
            @test increment_field!!(x, 3.0, 1) == (a=8.0, b=nt)
            @test increment_field!!(x, nt, 2) == (a=5.0, b=nt)
        end
        @testset "Tangent" begin
            nt = NoTangent()
            x = tangent((a=5.0, b=nt))
            @test @inferred(increment_field!!(x, 3.0, SSym(:a))) == tangent((a=8.0, b=nt))
            @test @inferred(increment_field!!(x, nt, SSym(:b))) == tangent((a=5.0, b=nt))
            @test @inferred(increment_field!!(x, 3.0, SInt(1))) == tangent((a=8.0, b=nt))
            @test @inferred(increment_field!!(x, nt, SInt(2))) == tangent((a=5.0, b=nt))

            # Slow versions.
            @test increment_field!!(x, 3.0, :a) == tangent((a=8.0, b=nt))
            @test increment_field!!(x, nt, :b) == tangent((a=5.0, b=nt))
            @test increment_field!!(x, 3.0, 1) == tangent((a=8.0, b=nt))
            @test increment_field!!(x, nt, 2) == tangent((a=5.0, b=nt))
        end
        @testset "MutableTangent" begin
            nt = NoTangent()
            @testset "$f" for (f, val, comp) in [
                (SSym(:a), 3.0, mutable_tangent((a=8.0, b=nt))),
                (:a, 3.0, mutable_tangent((a=8.0, b=nt))),
                (SSym(:b), nt, mutable_tangent((a=5.0, b=nt))),
                (:b, nt, mutable_tangent((a=5.0, b=nt))),
                (SInt(1), 3.0, mutable_tangent((a=8.0, b=nt))),
                (1, 3.0, mutable_tangent((a=8.0, b=nt))),
                (SInt(2), nt, mutable_tangent((a=5.0, b=nt))),
                (2, nt, mutable_tangent((a=5.0, b=nt))),
            ]
                x = mutable_tangent((a=5.0, b=nt))
                @test @inferred(increment_field!!(x, val, f)) == comp
                @test @inferred(increment_field!!(x, val, f)) === x
            end
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
        @test set_field_to_zero!!(x, 1) == mutable_tangent(nt2)
        @test set_field_to_zero!!(x, 1) === x
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
