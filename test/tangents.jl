@testset "tangents" begin

    # Each tuple is of the form (primal, t1, t2, increment!!(t1, t2)).
    @testset "$(typeof(p))" for (p, x, y, z) in vcat(
        [
            (sin, Tangent((;)), Tangent((;)), Tangent((;))),
            map(Float16, (5.0, 4.0, 3.1, 7.1)),
            (5f0, 4f0, 3f0, 7f0),
            (5.1, 4.0, 3.0, 7.0),
            ([3.0, 2.0], [1.0, 2.0], [2.0, 3.0], [3.0, 5.0]),
            (
                [1, 2],
                [NoTangent(), NoTangent()],
                [NoTangent(), NoTangent()],
                [NoTangent(), NoTangent()],
            ),
            (
                [[1.0], [1.0, 2.0]],
                [[2.0], [2.0, 3.0]],
                [[3.0], [4.0, 5.0]],
                [[5.0], [6.0, 8.0]],
            ),
            (
                setindex!(Vector{Vector{Float64}}(undef, 2), [1.0], 1),
                setindex!(Vector{Vector{Float64}}(undef, 2), [2.0], 1),
                setindex!(Vector{Vector{Float64}}(undef, 2), [3.0], 1),
                setindex!(Vector{Vector{Float64}}(undef, 2), [5.0], 1),
            ),
            (
                setindex!(Vector{Vector{Float64}}(undef, 2), [1.0], 2),
                setindex!(Vector{Vector{Float64}}(undef, 2), [2.0], 2),
                setindex!(Vector{Vector{Float64}}(undef, 2), [3.0], 2),
                setindex!(Vector{Vector{Float64}}(undef, 2), [5.0], 2),
            ),
            (
                (6.0, [1.0, 2.0]),
                (5.0, [3.0, 4.0]),
                (4.0, [4.0, 3.0]),
                (9.0, [7.0, 7.0]),
            ),
            (
                (a=6.0, b=[1.0, 2.0]),
                (a=5.0, b=[3.0, 4.0]),
                (a=4.0, b=[4.0, 3.0]),
                (a=9.0, b=[7.0, 7.0]),
            ),
            ((;), (;), (;), (;)),
            (
                TypeStableMutableStruct{Float64}(5.0, 3.0),
                build_tangent(TypeStableMutableStruct{Float64}, 5.0, 4.0),
                build_tangent(TypeStableMutableStruct{Float64}, 3.0, 3.0),
                build_tangent(TypeStableMutableStruct{Float64}, 8.0, 7.0),
            ),
            ( # complete init
                StructFoo(6.0, [1.0, 2.0]),
                build_tangent(StructFoo, 5.0, [3.0, 4.0]),
                build_tangent(StructFoo, 3.0, [2.0, 1.0]),
                build_tangent(StructFoo, 8.0, [5.0, 5.0]),
            ),
            ( # partial init
                StructFoo(6.0),
                build_tangent(StructFoo, 5.0),
                build_tangent(StructFoo, 4.0),
                build_tangent(StructFoo, 9.0),
            ),
            ( # complete init
                MutableFoo(6.0, [1.0, 2.0]),
                build_tangent(MutableFoo, 5.0, [3.0, 4.0]),
                build_tangent(MutableFoo, 3.0, [2.0, 1.0]),
                build_tangent(MutableFoo, 8.0, [5.0, 5.0]),
            ),
            ( # partial init
                MutableFoo(6.0),
                build_tangent(MutableFoo, 5.0),
                build_tangent(MutableFoo, 4.0),
                build_tangent(MutableFoo, 9.0),
            ),
            (
                UnitRange{Int}(5, 7),
                build_tangent(UnitRange{Int}, NoTangent(), NoTangent()),
                build_tangent(UnitRange{Int}, NoTangent(), NoTangent()),
                build_tangent(UnitRange{Int}, NoTangent(), NoTangent()),
            ),
        ],
        map([
            LowerTriangular{Float64, Matrix{Float64}},
            UpperTriangular{Float64, Matrix{Float64}},
            UnitLowerTriangular{Float64, Matrix{Float64}},
            UnitUpperTriangular{Float64, Matrix{Float64}},
        ]) do T
            return (
                T(randn(2, 2)),
                build_tangent(T, [1.0 2.0; 3.0 4.0]),
                build_tangent(T, [2.0 1.0; 5.0 4.0]),
                build_tangent(T, [3.0 3.0; 8.0 8.0]),
            )
        end,
        [
            (p, NoTangent(), NoTangent(), NoTangent()) for p in
                [Array, Float64, Union{Float64, Float32}, Union, UnionAll,
                Core.Intrinsics.xor_int, typeof(<:)]
        ],
    )
        rng = Xoshiro(123456)
        test_tangent(rng, p, z, x, y)
        test_numerical_testing_interface(p, x)
    end

    tangent(nt::NamedTuple) = Tangent(map(PossiblyUninitTangent, nt))
    mutable_tangent(nt::NamedTuple) = MutableTangent(map(PossiblyUninitTangent, nt))

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
