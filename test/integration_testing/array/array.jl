using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using LinearAlgebra, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule

sr(n::Int) = StableRNG(n)

_getter() = 5.0
@testset "array" begin
    test_cases = vcat(
        Any[
            (false, :allocs, adjoint, randn(sr(1), 5)),
            (false, :allocs, adjoint, randn(sr(2), 5, 4)),
            (false, :none, *, 2.1, randn(sr(3), 2)),
            (false, :none, *, -2.01, randn(sr(4), 1, 2)),
            (false, :none, *, -0.5, randn(sr(5), 2, 2, 1)),
            (false, :none, *, randn(sr(6), 2), 0.32),
            (false, :none, *, randn(sr(7), 2, 1), 0.33),
            (false, :none, *, randn(sr(8), 2, 1, 2), -0.34),
            (false, :none, *, randn(sr(9), 2, 2), randn(sr(12), 2)),
            (false, :none, *, randn(sr(10), 2, 2), randn(sr(11), 2, 3)),
            (false, :none, *, 4.4, randn(sr(15), 3, 2), randn(sr(13), 2, 2), [-0.5, 0.1]),
            (false, :none, *, 4.3, randn(sr(16), 2), adjoint(randn(sr(17), 2, 1))),
            (
                false,
                :none,
                *,
                randn(sr(19)),
                randn(sr(20), 2),
                transpose(randn(sr(18), 2, 1)),
            ),
            (false, :none, *, adjoint(randn(sr(22), 2, 2)), randn(sr(21), 2)),
            (false, :none, *, Diagonal(randn(sr(23), 2)), randn(sr(24), 2)),
            # (false, :none, , *, randn(sr(27), 2)', Diagonal(randn(sr(26), 2)), randn(sr(25), 2)), https://github.com/compintell/Mooncake.jl/issues/319
            (false, :none, *, randn(sr(28), 2, 3)', randn(sr(29), 2)),
            (false, :none, *, 4.0 * I, randn(sr(30), 2)),
            (false, :none, *, 3.5 * I, randn(sr(31), 2, 3)),
            (false, :none, *, UpperTriangular(randn(sr(31), 2, 2)), randn(sr(32), 2)),
            (false, :none, *, LowerTriangular(randn(sr(34), 2, 2)), randn(sr(33), 2)),
            (false, :none, *, transpose(randn(sr(36), 2, 3)), randn(sr(35), 2)),
            (
                false,
                :none,
                *,
                randn(sr(37), 2)',
                adjoint(randn(sr(38), 3, 2)),
                randn(sr(39), 3),
            ),
            (
                false,
                :none,
                *,
                randn(sr(42), 2)',
                transpose(randn(sr(41), 3, 2)),
                randn(sr(40), 3),
            ),
            (false, :none, *, randn(sr(43), 1), transpose(randn(sr(44), 2, 1))),
            (
                false,
                :none,
                *,
                transpose(randn(sr(47), 2, 3)),
                Diagonal(randn(sr(46), 2)),
                randn(sr(45), 2),
            ),
            (false, :none, *, adjoint(randn(sr(46), 2, 3)), randn(sr(47), 2)),
            (false, :none, *, transpose(randn(sr(49), 2, 3)), randn(sr(48), 2)),
            (
                false,
                :none,
                *,
                adjoint(randn(sr(50), 2, 1)),
                randn(sr(51), 2, 3),
                randn(sr(52), 3),
            ),
            (
                false,
                :none,
                *,
                transpose(randn(sr(55), 2, 1)),
                randn(sr(54), 2, 3),
                randn(sr(53), 3),
            ),
            (
                false,
                :none,
                *,
                adjoint(randn(sr(55), 2, 1)),
                randn(sr(1), 2, 3),
                randn(sr(2), 3),
                randn(sr(3)),
            ),
            (
                false,
                :none,
                *,
                transpose(randn(sr(4), 2, 1)),
                randn(sr(5), 2, 3),
                randn(sr(6), 3),
                randn(sr(7)),
            ),
            (
                false,
                :none,
                *,
                adjoint(randn(sr(8), 2, 2)),
                randn(sr(9), 2, 3),
                randn(sr(10), 3, 2),
                randn(sr(11), 2),
            ),
            (
                false,
                :none,
                *,
                transpose(randn(sr(1), 2, 2)),
                randn(sr(2), 2, 3),
                randn(sr(3), 3, 2),
                randn(sr(4), 2),
            ),
            (false, :none, *, transpose(randn(sr(5), 2, 3)), randn(sr(6), 2)),
            (false, :none, *, randn(sr(8), 1), adjoint(randn(sr(7), 2, 1))),
            (false, :none, *, randn(sr(9), 1), transpose(randn(sr(0), 2, 1))),
            (
                false,
                :none,
                *,
                randn(sr(3), 1),
                adjoint(randn(sr(2), 2, 1)),
                randn(sr(1), 2, 2),
            ),
            (
                false,
                :none,
                *,
                randn(sr(4), 1),
                transpose(randn(sr(5), 2, 1)),
                randn(sr(6), 2, 2),
            ),
            (false, :none, *, randn(sr(9), 1), adjoint(randn(sr(8), 2, 1)), randn(sr(7))),
            (false, :none, *, randn(sr(0), 1), transpose(randn(sr(1), 2, 1)), randn(sr(2))),
            (false, :none, *, randn(sr(4), 1), adjoint(randn(sr(3), 2, 1))),
            (false, :none, *, randn(sr(5), 1), transpose(randn(sr(6), 2, 1))),
            (false, :none, *, randn(sr(9), 3, 2), randn(sr(8), 2, 2), randn(sr(7), 2)),
            (false, :none, *, randn(sr(0), 1), randn(sr(1), 1, 3)),
            (false, :none, *, randn(sr(3), 3, 2), randn(sr(2), 2)),
            (false, :none, *, randn(sr(4), 3, 2), randn(sr(5), 2), randn(sr(6))),
            (
                false,
                :none,
                *,
                randn(sr(0), 3, 2),
                randn(sr(9), 2),
                randn(sr(8)),
                randn(sr(7)),
            ),
            (
                false,
                :none,
                *,
                randn(sr(1), 2, 2),
                randn(sr(2), 2, 3),
                randn(sr(3), 3),
                randn(sr(4)),
            ),
            (
                false,
                :none,
                *,
                randn(sr(5), 2, 1),
                randn(sr(6), 1, 2),
                randn(sr(7), 2, 3),
                randn(sr(8), 3),
            ),
            (false, :none, *, randn(sr(9), 2), 5.0 * I),
            (
                false,
                :none,
                *,
                randn(sr(0)),
                adjoint(randn(sr(1), 2, 1)),
                randn(sr(2), 2, 3),
                randn(sr(3), 3),
            ),
            (
                false,
                :none,
                *,
                randn(sr(4)),
                transpose(randn(sr(5), 2, 1)),
                randn(sr(6), 2, 3),
                randn(sr(7), 3),
            ),
            (
                false,
                :none,
                *,
                randn(sr(8)),
                randn(sr(9)),
                randn(sr(0), 3, 2),
                randn(sr(1), 2),
            ),
            (false, :none, *, randn(sr(2), 2)', randn(sr(3), 2, 3), randn(sr(4), 3)),
            (false, :none, +, randn(sr(5), 2), randn(sr(6), 2)),
            (
                false,
                :none,
                +,
                randn(sr(7), 2),
                randn(sr(8), 2),
                randn(sr(9), 2),
                randn(sr(0), 2),
            ),
            (false, :none, -, randn(sr(1), 1, 2), randn(sr(2), 1, 2)),
            (false, :none, -, randn(sr(3), 1, 2)),
            (false, :none, /, randn(sr(6), 1, 2), 0.66),
            (false, :none, /, randn(sr(7), 2), 5.0 * I),
            (false, :none, /, randn(sr(8), 1), Diagonal(rand(sr(9), 1) .+ 1)),
            (false, :none, /, randn(sr(0), 2, 2), Diagonal(rand(sr(1), 2) .+ 1)),
        ],
        map([
            rand(sr(1), 2, 2) + I,
            Bidiagonal(rand(sr(2), 3) .+ 1, [0.3, 0.6], 'L'),
            Bidiagonal(rand(sr(3), 3) .+ 1, [0.4, -0.3], 'U'),
            Diagonal(rand(sr(4), 2) .+ 1),
            LowerTriangular(randn(sr(5), 2, 2) + I),
            UpperTriangular(randn(sr(6), 2, 2) + I),
            UnitLowerTriangular(randn(sr(7), 2, 2)),
            UnitUpperTriangular(randn(sr(8), 2, 2)),
        ]) do X
            (false, :none, /, randn(sr(1), 2, size(X, 1)), X)
        end,
        Any[
            (false, :none, //, [1, 2], 5),
            (false, :allocs, ==, randn(sr(1), 1, 2), randn(2, 1)),
            (false, :none, Array, randn(sr(2), 1, 2)),
            (false, :allocs, Bidiagonal, randn(sr(3), 4), randn(sr(4), 3), 'L'),
            (false, :allocs, CartesianIndices, [1, 2, 3]),
            (false, :allocs, Diagonal, randn(sr(5), 4)),
            (false, :none, IndexStyle, randn(sr(6), 1, 2)),
            (false, :none, IndexStyle, randn(sr(7), 2, 1), randn(sr(8), 1, 2)),
            (false, :none, LinearIndices, randn(sr(9), 3)),
            (false, :none, LinearIndices, randn(sr(0), 3, 2)),
            (false, :none, PermutedDimsArray, randn(sr(1), 2, 3), [2, 1]),
            (false, :none, SubArray, randn(sr(2), 2, 3), (1:2, 2:3)),
            (false, :none, \, 0.67, randn(sr(3), 2, 2)),
            # (false, :none, \, Hermitian(randn(sr(0), 2, 2) + 3I), randn(sr(1), 2)), # missing foreigncall rule
            # (false, :none, \, Symmetric(randn(sr(3), 2, 2) + 3I), randn(sr(2), 2)), # missing foreigncall rule
            # (false, :none, \, SymTridiagonal(rand(sr(4), 3) .+ 1, randn(sr(5), 2)), randn(sr(6), 3)), # missing foreigncall
            (false, :none, \, 3.0 * I, randn(sr(4), 2)),
            (false, :none, \, 3.0 * I, randn(sr(5), 2, 1)),
            (false, :none, \, UnitLowerTriangular(randn(sr(6), 2, 2)), randn(sr(7), 2)),
            (false, :none, \, UnitUpperTriangular(randn(sr(9), 2, 2)), randn(sr(8), 2)),
            (
                false,
                :none,
                (A, x) -> cholesky(A)' \ x,
                Symmetric(randn(sr(0), 2, 2) + 5I),
                randn(sr(1), 2),
            ),
            (
                false,
                :none,
                (A, X) -> cholesky(A)' \ X,
                Symmetric(randn(sr(2), 2, 2) + 5I),
                randn(sr(3), 2, 3),
            ),
            (
                false,
                :none,
                \,
                adjoint(Bidiagonal(rand(sr(4), 3) .+ 1, randn(sr(5), 2), 'U')),
                randn(sr(6), 3),
            ),
            (
                false,
                :none,
                \,
                adjoint(Bidiagonal(rand(sr(8), 3) .+ 1, randn(sr(9), 2), 'U')),
                randn(sr(7), 3, 2),
            ),
            (
                false,
                :none,
                \,
                adjoint(Bidiagonal(rand(sr(9), 3) .+ 1, randn(sr(0), 2), 'L')),
                randn(sr(8), 3),
            ),
            (
                false,
                :none,
                \,
                adjoint(Bidiagonal(rand(sr(0), 3) .+ 1, randn(sr(1), 2), 'L')),
                randn(sr(9), 3, 2),
            ),
            (false, :none, \, randn(sr(2), 2), randn(sr(3), 2, 2)),
            (false, :none, \, LowerTriangular(rand(sr(4), 2, 2) + 3I), randn(sr(5), 2)),
            (false, :none, \, UpperTriangular(rand(sr(7), 2, 2) + 3I), randn(sr(6), 2)),
            (false, :none, \, Diagonal(rand(sr(9), 2) .+ 1), randn(sr(8), 2)),
            (
                false,
                :none,
                \,
                Bidiagonal(rand(sr(0), 3) .+ 1, randn(sr(1), 2), 'U'),
                randn(sr(2), 3),
            ),
            (
                false,
                :none,
                \,
                Bidiagonal(rand(sr(4), 3) .+ 1, randn(sr(5), 2), 'U'),
                randn(sr(6), 3, 2),
            ),
            (
                false,
                :none,
                \,
                Bidiagonal(rand(sr(7), 3) .+ 1, randn(sr(8), 2), 'L'),
                randn(sr(9), 3),
            ),
            (
                false,
                :none,
                \,
                Bidiagonal(rand(sr(1), 3) .+ 1, randn(sr(2), 2), 'L'),
                randn(sr(0), 3, 2),
            ),
            (false, :none, \, rand(sr(3), 2, 2) + 3I, randn(sr(4), 2)),
            (false, :none, \, rand(sr(6), 2, 2) + 3I, randn(sr(5), 2, 3)),
            (false, :allocs, x -> all(>(0), x), randn(sr(7), 2)),
            (false, :allocs, allunique, randn(sr(8), 2)),
            (false, :allocs, x -> any(<(0), x), randn(sr(9), 2)),
            (false, :none, append!, randn(sr(1), 3), randn(sr(0), 2)),
            (false, :allocs, argmax, randn(sr(2), 2)),
            (false, :allocs, argmin, randn(sr(3), 2)),
            (false, :allocs, axes, randn(sr(4), 2, 1, 1, 2)),
            (false, :none, circshift, randn(sr(5), 3), 1),
            (false, :allocs, circshift!, randn(sr(6), 3), randn(sr(7), 3), 1),
            (false, :allocs, clamp!, randn(sr(8), 3), 0.0, 0.5),
            (false, :none, collect, randn(sr(9), 3)),
            (false, :none, complex, randn(sr(1), 2)),
            (false, :none, conj, randn(sr(2), 3)),
            (false, :none, copy!, randn(sr(3), 2), randn(sr(4), 2)),
            (
                false,
                :none,
                copyto!,
                randn(sr(5), 3),
                CartesianIndices(2:3),
                randn(sr(6), 2),
                CartesianIndices(1:2),
            ),
            (false, :none, copyto!, randn(sr(7), 3), 2, randn(sr(8), 2), 1, 2),
            (false, :none, copyto!, randn(sr(1), ComplexF64, 3), 2, randn(sr(12), 2), 1, 2),
            (false, :none, copyto!, randn(sr(1), 3), randn(sr(2), 3)),
            (false, :none, copyto!, randn(sr(4), ComplexF64, 1, 3), randn(sr(5), 1, 3)),
            (
                false,
                :none,
                copyto!,
                PermutedDimsArray(randn(sr(7), 2, 3), [2, 1]),
                randn(sr(6), 3, 2),
            ),
            (false, :allocs, x -> count(<(0), x), randn(sr(7), 2, 3)),
            (
                false,
                :allocs,
                (r, A) -> count!(<(0), r, A),
                randn(sr(8), 2),
                randn(sr(9), 2, 3),
            ),
            (false, :none, cumprod, randn(sr(0), 3)),
            (
                false,
                :none,
                (B, A) -> cumprod!(B, A; dims=2),
                randn(sr(1), 2, 2),
                randn(sr(2), 2, 2),
            ),
            (false, :none, cumsum, randn(sr(3), 3)),
            (
                false,
                :allocs,
                (out, v) -> cumsum!(out, v; dims=1),
                randn(sr(4), 3),
                randn(sr(5), 3),
            ),
            (
                false,
                :none,
                (out, v) -> cumsum!(out, v; dims=2),
                randn(sr(7), 2, 3),
                randn(sr(6), 2, 3),
            ),
            (false, :none, deleteat!, randn(sr(8), 5), 3),
            (false, :none, deleteat!, randn(sr(9), 5), 1),
            (false, :none, deleteat!, randn(sr(0), 5), 5),
            (false, :none, deleteat!, randn(sr(1), 5), 2:3),
            (false, :none, deleteat!, randn(sr(2), 3), [false, true, true]),
            (false, :none, deleteat!, randn(sr(3), 5), [1, 2, 5]),
            (false, :none, diff, randn(sr(4), 5)),
            (false, :none, x -> diff(x; dims=1), randn(sr(5), 3, 2)),
            (false, :none, x -> diff(x; dims=2), randn(sr(6), 3, 2)),
            (false, :none, empty, randn(sr(7), 5)),
            (false, :allocs, extrema, randn(sr(8), 3)),
            (false, :allocs, x -> extrema(sin, x), randn(sr(9), 3)),
            (false, :none, x -> extrema(sin, x; dims=1), randn(sr(0), 3, 2)),
            (
                false,
                :none,
                (r, A) -> extrema!(cos, r, A),
                tuple.(randn(sr(1), 3), randn(sr(2), 3)),
                randn(sr(3), 3, 2),
            ),
            (
                false,
                :none,
                (r, A) -> extrema!(r, A),
                tuple.(randn(sr(4), 3), randn(sr(5), 3)),
                randn(sr(6), 3, 2),
            ),
            (false, :allocs, fill!, randn(sr(7), 3), randn(sr(8))),
            (false, :allocs, fill!, randn(sr(0), 3, 2), randn(sr(9))),
            (false, :none, x -> filter(>(0), x), [0.5, -0.1, -0.4]),
            (false, :none, x -> filter(<(0), x), randn(sr(1), 2, 2)),
            (false, :none, x -> findall(<(0), x), [0.5, 0.0, -0.3]),
            (false, :allocs, x -> findfirst(<(0), x), [0.5, -0.1, -0.4]),
            (false, :allocs, x -> findlast(<(0), x), [0.5, -0.1, -0.4]),
            (false, :none, findmax, randn(sr(1), 2, 2)),
            (false, :none, x -> findmax(sin, x), [0.5, -0.1, -0.4]),
            (false, :none, x -> findmax(sin, x), [0.5, -0.1, -0.4]),
            (false, :none, findmin, randn(sr(2), 2, 2)),
            (false, :none, x -> findmin(cos, x), randn(sr(3), 2, 2)),
            (false, :allocs, first, randn(sr(4), 3)),
            (false, :allocs, firstindex, randn(sr(5), 3)),
            (false, :none, float, randn(sr(6), 3)),
            (false, :none, (x, i) -> get(_getter, x, i), randn(sr(7), 3), 2),
            (false, :none, (x, i) -> get(_getter, x, i), randn(sr(8), 3), 4),
            (false, :allocs, getindex, randn(sr(9), 5), 1),
            (false, :allocs, getindex, randn(sr(0), 5), 3),
            (false, :allocs, getindex, randn(sr(1), 5, 4), 3),
            (false, :allocs, getindex, randn(sr(2), 5, 4), 2, 3),
            (false, :none, getindex, randn(sr(3), 5), 1:2),
            (false, :none, getindex, randn(sr(4), 5), 2:4),
            (false, :none, getindex, randn(sr(5), 5), :),
            (false, :none, getindex, randn(sr(6), 3, 2), :),
            (false, :none, getindex, randn(sr(7), 5), 1:2:5),
            (false, :none, getindex, randn(sr(8), 5), [1, 3, 5]),
            (false, :none, hash, randn(sr(9), 3, 4), UInt(2)),
            (false, :none, hcat, randn(sr(0), 3, 2)),
            (false, :none, hcat, randn(sr(1), 2, 3), randn(sr(2), 2, 2)),
            (false, :none, imag, randn(sr(3), 2, 3)),
            (false, :none, imag, randn(sr(4), ComplexF64, 2, 3)),
            (false, :none, insert!, randn(sr(5), 3), 2, randn(sr(6))),
            (false, :none, isassigned, randn(sr(7), 5), 3),
            (false, :none, isassigned, randn(sr(8), 5), CartesianIndex(4)),
            (false, :none, isempty, randn(sr(9), 5)),
            (false, :none, isempty, randn(sr(0), 0)),
            (false, :none, isequal, randn(sr(1), 3), randn(sr(2), 3)),
            (false, :none, isreal, randn(sr(3), 3)),
            (false, :none, isreal, randn(sr(4), ComplexF64, 3)),
            (false, :none, iszero, randn(sr(5), 2)),
            (false, :none, iterate, randn(sr(6), 2)),
            (false, :none, iterate, randn(sr(7), 2, 2)),
            (false, :none, keys, randn(sr(8), 2)),
            (false, :none, keytype, randn(sr(9), 2)),
            (false, :none, kron, randn(sr(0)), randn(sr(1), 2, 2)),
            (false, :none, kron, randn(sr(3)), randn(sr(2), 2)),
            (false, :none, kron, randn(sr(4), 2, 2), randn(sr(5))),
            (false, :none, kron, randn(sr(7), 2), randn(sr(6))),
            (false, :none, kron, randn(sr(8), 2), rand(sr(9), 3)),
            (false, :none, kron, randn(sr(1), 2), randn(sr(0), 1, 2)),
            (false, :none, kron, randn(sr(2), 2, 1), randn(sr(3), 3)),
            (false, :none, kron, randn(sr(5), 2, 2), randn(sr(4), 2, 2)),
            (false, :none, kron!, randn(sr(6), 2), randn(sr(7), 2), randn(sr(8))),
            (false, :none, kron!, randn(sr(1), 2, 2), randn(sr(0), 2, 2), randn(sr(9))),
            (false, :none, kron!, randn(sr(2), 2), randn(sr(3)), randn(sr(4), 2)),
            (false, :none, kron!, randn(sr(7), 2, 2), randn(sr(6)), randn(sr(5), 2, 2)),
            (false, :none, kron!, randn(sr(8), 6), randn(sr(9), 2), randn(sr(0), 3)),
            (false, :none, kron!, randn(sr(3), 4, 3), randn(sr(2), 2), randn(sr(1), 2, 3)),
            (
                false,
                :none,
                kron!,
                randn(sr(4), 4, 6),
                randn(sr(5), 2, 2),
                randn(sr(6), 2, 3),
            ),
            (false, :none, kron!, randn(sr(9), 3, 2), randn(sr(8), 1, 2), randn(sr(7), 3)),
            (
                false,
                :none,
                kron!,
                randn(sr(0), 6, 2),
                randn(sr(1), 2, 2),
                randn(sr(3), 3, 1),
            ),
            (false, :none, last, randn(sr(4), 4)),
            (false, :none, lastindex, randn(sr(5), 2)),
            (false, :none, length, randn(sr(6), 2)),
            (false, :none, length, randn(sr(7), 2, 3)),
            (false, :none, map, sin, randn(sr(8), 2)),
            (false, :none, map!, sin, randn(sr(9), 2), randn(sr(0), 2)),
            (false, :none, map!, *, randn(sr(3), 2), randn(sr(2), 2), randn(sr(1), 2)),
            (false, :none, mapreduce, sin, *, randn(sr(4), 2)),
            (false, :none, (f, x) -> mapslices(f, x; dims=1), sum, randn(sr(5), 2, 3)),
            (false, :none, maximum, randn(sr(6), 2)),
            (false, :none, maximum, randn(sr(7), 2, 3)),
            (false, :none, maximum, sin, randn(sr(8), 2)),
            (false, :none, maximum, cos, randn(sr(9), 2, 3)),
            (false, :none, x -> maximum(cos, x; dims=2), randn(sr(0), 3, 2)),
            (false, :none, maximum!, sin, randn(sr(2), 2), randn(sr(1), 2, 3)),
            (false, :none, minimum, randn(sr(3), 2)),
            (false, :none, minimum, randn(sr(4), 2, 3)),
            (false, :none, minimum, sin, randn(sr(5), 2)),
            (false, :none, minimum, cos, randn(sr(6), 2, 3)),
            (false, :none, x -> minimum(cos, x; dims=2), randn(sr(7), 3, 2)),
            (false, :none, minimum!, sin, randn(sr(9), 2), randn(sr(8), 2, 3)),
        ],
        vec(
            reduce(
                vcat,
                map(
                    Iterators.product(
                        [adjoint(randn(sr(0), 2, 3)), transpose(randn(sr(1), 2, 3))],
                        [randn(sr(3), 2), randn(sr(2), 2, 3)],
                        [randn(sr(4)), randn(sr(5), 1), randn(sr(6), 3)],
                    ),
                ) do (A, b, z)
                    (false, :none, muladd, A, b, z)
                end,
            ),
        ),
        Any[
            (false, :none, ndims, randn(sr(7), 2)),
            (false, :none, ndims, randn(sr(8), 1, 2, 1, 1, 1)),
            (false, :none, nextind, randn(sr(9), 3, 3), 2),
            (false, :none, nextind, randn(sr(0), 3, 3), CartesianIndex(2, 2)),
            (false, :none, pairs, randn(sr(1), 2, 2)),
            (false, :none, parent, randn(sr(2), 3, 2)),
            (false, :none, parentindices, randn(sr(3), 3, 2)),
            (false, :none, permute!, randn(sr(5), 4), [2, 4, 3, 1]),
            (false, :none, permutedims, randn(sr(6), 3, 2, 1), [2, 1, 3]),
            (false, :none, permutedims, randn(sr(7), 2)),
            (false, :none, permutedims, randn(sr(8), 2, 3)),
            (
                false,
                :none,
                permutedims!,
                randn(sr(9), 2, 3, 1),
                randn(sr(0), 3, 2, 1),
                [2, 1, 3],
            ),
            (false, :none, pop!, randn(sr(1), 5)),
            (false, :none, popat!, randn(sr(2), 5), 1),
            (false, :none, popat!, randn(sr(4), 10), 5),
            (false, :none, popat!, randn(sr(4), 5), 5),
            (false, :none, popat!, randn(sr(5), 5), 7, 3.0),
            (false, :none, popfirst!, randn(sr(6), 5)),
            (false, :none, prepend!, randn(sr(7), 5), randn(sr(8), 3)),
            (false, :none, prevind, randn(sr(9), 2, 3), 5),
            (false, :none, prevind, randn(sr(0), 2, 3), CartesianIndex(2, 2)),
            (false, :none, prod, randn(sr(1), 2)),
            (false, :none, prod, randn(sr(2), 2, 3)),
            (false, :none, x -> prod(x; dims=1), randn(sr(3), 2, 3)),
            (false, :none, x -> prod(sin, x; dims=2), randn(sr(4), 2, 2)),
            (false, :none, prod!, sin, randn(sr(6), 2), randn(sr(5), 2, 3)),
            (false, :none, prod!, randn(sr(7), 2), randn(sr(8), 2, 3)),
            (false, :none, promote_shape, randn(sr(0), 2), randn(sr(9), 2)),
            (false, :none, promote_shape, randn(sr(1), 2), randn(sr(2), 2, 1, 1)),
            (false, :none, push!, randn(sr(3), 5), 4.0),
            (false, :none, pushfirst!, randn(sr(4), 5), 3.0),
            (false, :none, real, randn(sr(5), 3, 2, 1)),
            (false, :none, reduce, *, randn(sr(6), 2)),
            (false, :none, reduce, *, randn(sr(7), 3, 2)),
            (false, :none, repeat, randn(sr(8), 2), 2, 3),
            (false, :none, (x, i, o) -> repeat(x; inner=i, outer=o), randn(sr(9), 2), 2, 3),
            (false, :none, reshape, randn(sr(0), 5), 1, 5),
            (false, :none, reshape, randn(sr(0), 3, 2), 2, 3),
            (false, :none, reshape, randn(sr(1), 3, 2), :, 3),
            (false, :none, reshape, randn(sr(2), 3, 2), (6,)),
            (false, :none, x -> (resize!(x, 10); x[6:end] .= x[1:5]), randn(sr(3), 5)),
            (false, :none, reverse, randn(sr(4), 3)),
            (false, :none, reverse!, randn(sr(5), 3)),
            (false, :none, reverseind, randn(sr(6), 5), 3),
            (false, :none, selectdim, randn(sr(7), 3), 1, 2),
            (false, :none, setdiff!, randn(sr(8), 3), randn(sr(9), 2)),
            (false, :none, setindex!, randn(sr(0), 3), randn(sr(1)), 1),
            (false, :none, setindex!, randn(sr(2), 3), randn(sr(3), 2), 1:2),
            (false, :none, setindex!, randn(sr(4), 3), randn(sr(5)), CartesianIndex(2)),
            (false, :none, setindex!, randn(sr(7), 3, 2), randn(sr(6)), 5),
            (false, :none, setindex!, randn(sr(8), 3, 2), randn(sr(9)), 2, 2),
            (
                false,
                :none,
                setindex!,
                randn(sr(0), 3, 2),
                randn(sr(1)),
                CartesianIndex(2, 1),
            ),
            (false, :none, setindex!, randn(sr(2), 3, 2), randn(sr(3), 2, 2), 2:3, 1:2),
            (false, :none, setindex!, randn(sr(5), 3, 2), randn(sr(4), 2, 2), 1:2, :),
            (
                false,
                :none,
                setindex!,
                randn(sr(6), 3, 2),
                randn(sr(7)),
                CartesianIndex(2),
                1,
            ),
            (true, :none, similar, randn(sr(8), 3, 2)),
            (false, :none, size, randn(sr(9), 2)),
            (false, :none, size, randn(sr(0), 3, 2)),
            (false, :none, size, randn(sr(1), 3, 2, 1)),
            (false, :none, sizeof, randn(sr(2), 5)),
            (false, :none, sort, randn(sr(3), 3)),
            (false, :none, sort!, randn(sr(4), 3)),
            (false, :none, sortperm, randn(sr(0), 3)),
            (false, :none, sortperm!, [1, 2, 3], randn(sr(1), 3)),
            (false, :none, x -> sortslices(x; dims=1), randn(sr(2), 2, 3)),
            (false, :none, splice!, randn(sr(5), 5), 1:2),
            (false, :none, splice!, randn(sr(6), 5), 1:0),
            (false, :none, splice!, randn(sr(7), 5), 1:5),
            (false, :none, splice!, randn(sr(8), 5), 1:2, randn(sr(6), 3)),
            (false, :none, splice!, randn(sr(9), 5), 3:2, randn(sr(5), 4)),
            (false, :none, splice!, randn(sr(0), 5), 1),
            (false, :none, splice!, randn(sr(1), 5), 1, randn(sr(3))),
            (false, :none, splice!, randn(sr(2), 5), 1, randn(sr(4), 3)),
            (false, :none, stride, randn(sr(7), 3, 2), 1),
            (false, :none, sum, randn(sr(8), 2, 3)),
            (false, :none, x -> sum(x; dims=1), randn(sr(9), 3, 2)),
            (false, :none, (f, x) -> sum(f, x; dims=2), sin, randn(sr(0), 3, 2)),
            (false, :none, sum!, randn(sr(2), 1, 3), randn(sr(1), 2, 3)),
            (false, :none, sum!, sin, randn(sr(3), 1, 3), randn(sr(4), 2, 3)),
            (false, :none, transpose, randn(sr(5), 3)),
            (false, :none, transpose, randn(sr(6), 3, 2)),
            (false, :none, transpose, randn(sr(7), 1, 3)),
            (false, :none, unique, randn(sr(8), 3)),
            (false, :none, valtype, randn(sr(9), 3)),
            (false, :none, vcat, randn(sr(0), 2)),
            (false, :none, vcat, randn(sr(1), 2, 2)),
            (false, :none, vcat, randn(sr(2), 3), randn(2)),
            (false, :none, vcat, randn(sr(3), 3, 2), randn(sr(4), 2, 2)),
            (false, :none, vec, randn(sr(5), 2)),
            (false, :none, vec, randn(sr(6), 2, 3)),
            (false, :none, view, randn(sr(7), 2, 3), 1:2, 1:2),
            (false, :none, view, randn(sr(8), 3, 2), 1:3),
            (false, :none, view, randn(sr(9), 3, 2), :, 1),
            (false, :none, view, randn(sr(0), 3, 2), :, :),
            (false, :none, zero, randn(sr(1), 3)),
            (false, :none, zero, randn(sr(2), 2, 3)),
        ],
    )
    @testset for (interface_only, perf_flag, f, x...) in test_cases
        @info Mooncake._typeof((f, x...))
        test_rule(
            sr(123456),
            f,
            x...;
            interface_only,
            is_primitive=false,
            debug_mode=false,
            perf_flag,
        )
    end
end
