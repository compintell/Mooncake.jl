@testset "array" begin
    interp = Taped.TInterp()
    Taped.flush_interpreted_function_cache!()
    @testset for (interface_only, f, x...) in vcat(
        [
            (false, adjoint, randn(sr(1), 5)),
            (false, adjoint, randn(sr(2), 5, 4)),
            (false, *, 2.1, randn(sr(3), 2)),
            (false, *, -2.01, randn(sr(4), 1, 2)),
            (false, *, -0.5, randn(sr(5), 2, 2, 1)),
            (false, *, randn(sr(6), 2), 0.32),
            (false, *, randn(sr(7), 2, 1), 0.33),
            (false, *, randn(sr(8), 2, 1, 2), -0.34),
            (false, *, randn(sr(9), 2, 2), randn(sr(12), 2)),
            (false, *, randn(sr(10), 2, 2), randn(sr(11), 2, 3)),
            (false, *, 4.4, randn(sr(15), 3, 2), randn(sr(13), 2, 2), [-0.5, 0.1]),
            (false, *, 4.3, randn(sr(16), 2), adjoint(randn(sr(17), 2, 1))),
            (false, *, randn(sr(19)), randn(sr(20), 2), transpose(randn(sr(18), 2, 1))),
            (false, *, adjoint(randn(sr(22), 2, 2)), randn(sr(21), 2)),
            (false, *, Diagonal(randn(sr(23), 2)), randn(sr(24), 2)),
            (false, *, randn(sr(27), 2)', Diagonal(randn(sr(26), 2)), randn(sr(25), 2)),
            (false, *, randn(sr(28), 2, 3)', randn(sr(29), 2)),
            (false, *, 4.0 * I, randn(sr(30), 2)),
            (false, *, 3.5 * I, randn(sr(31), 2, 3)),
            (false, *, UpperTriangular(randn(sr(31), 2, 2)), randn(sr(32), 2)),
            (false, *, LowerTriangular(randn(sr(34), 2, 2)), randn(sr(33), 2)),
            (false, *, transpose(randn(sr(36), 2, 3)), randn(sr(35), 2)),
            (false, *, randn(sr(37), 2)', adjoint(randn(sr(38), 3, 2)), randn(sr(39), 3)),
            (false, *, randn(sr(42), 2)', transpose(randn(sr(41), 3, 2)), randn(sr(40), 3)),
            (false, *, randn(sr(43), 1), transpose(randn(sr(44), 2, 1))),
            (
                false,
                *,
                transpose(randn(sr(47), 2, 3)),
                Diagonal(randn(sr(46), 2)),
                randn(sr(45), 2),
            ),
            (false, *, adjoint(randn(sr(46), 2, 3)), randn(sr(47), 2)),
            (false, *, transpose(randn(sr(49), 2, 3)), randn(sr(48), 2)),
            (false, *, adjoint(randn(sr(50), 2, 1)), randn(sr(51), 2, 3), randn(sr(52), 3)),
            (
                false,
                *,
                transpose(randn(sr(55), 2, 1)),
                randn(sr(54), 2, 3),
                randn(sr(53), 3),
            ),
            (
                false,
                *,
                adjoint(randn(sr(55), 2, 1)),
                randn(sr(1), 2, 3),
                randn(sr(2), 3),
                randn(sr(3)),
            ),
            (
                false,
                *,
                transpose(randn(sr(4), 2, 1)),
                randn(sr(5), 2, 3),
                randn(sr(6), 3),
                randn(sr(7)),
            ),
            (
                false,
                *,
                adjoint(randn(sr(8), 2, 2)),
                randn(sr(9), 2, 3),
                randn(sr(10), 3, 2),
                randn(sr(11), 2),
            ),
            (
                false,
                *,
                transpose(randn(sr(1), 2, 2)),
                randn(sr(2), 2, 3),
                randn(sr(3), 3, 2),
                randn(sr(4), 2),
            ),
            (false, *, transpose(randn(sr(5), 2, 3)), randn(sr(6), 2)),
            (false, *, randn(sr(8), 1), adjoint(randn(sr(7), 2, 1))),
            (false, *, randn(sr(9), 1), transpose(randn(sr(0), 2, 1))),
            (false, *, randn(sr(3), 1), adjoint(randn(sr(2), 2, 1)), randn(sr(1), 2, 2)),
            (false, *, randn(sr(4), 1), transpose(randn(sr(5), 2, 1)), randn(sr(6), 2, 2)),
            (false, *, randn(sr(9), 1), adjoint(randn(sr(8), 2, 1)), randn(sr(7))),
            (false, *, randn(sr(0), 1), transpose(randn(sr(1), 2, 1)), randn(sr(2))),
            (false, *, randn(sr(4), 1), adjoint(randn(sr(3), 2, 1))),
            (false, *, randn(sr(5), 1), transpose(randn(sr(6), 2, 1))),
            (false, *, randn(sr(9), 3, 2), randn(sr(8), 2, 2), randn(sr(7), 2)),
            (false, *, randn(sr(0), 1), randn(sr(1), 1, 3)),
            (false, *, randn(sr(3), 3, 2), randn(sr(2), 2)),
            (false, *, randn(sr(4), 3, 2), randn(sr(5), 2), randn(sr(6))),
            (false, *, randn(sr(0), 3, 2), randn(sr(9), 2), randn(sr(8)), randn(sr(7))),
            (
                false,
                *,
                randn(sr(1), 2, 2),
                randn(sr(2), 2, 3),
                randn(sr(3), 3),
                randn(sr(4)),
            ),
            (
                false,
                *,
                randn(sr(5), 2, 1),
                randn(sr(6), 1, 2),
                randn(sr(7), 2, 3),
                randn(sr(8), 3),
            ),
            (false, *, randn(sr(9), 2), 5.0 * I),
            (
                false,
                *,
                randn(sr(0), ),
                adjoint(randn(sr(1), 2, 1)),
                randn(sr(2), 2, 3),
                randn(sr(3), 3),
            ),
            (
                false,
                *,
                randn(sr(4)),
                transpose(randn(sr(5), 2, 1)),
                randn(sr(6), 2, 3),
                randn(sr(7), 3),
            ),
            (false, *, randn(sr(8)), randn(sr(9)), randn(sr(0), 3, 2), randn(sr(1), 2)),
            (false, *, randn(sr(2), 2)', randn(sr(3), 2, 3), randn(sr(4), 3)),
            (false, +, randn(sr(5), 2), randn(sr(6), 2)),
            (false, +, randn(sr(7), 2), randn(sr(8), 2), randn(sr(9), 2), randn(sr(0), 2)),
            (false, -, randn(sr(1), 1, 2), randn(sr(2), 1, 2)),
            (false, -, randn(sr(3), 1, 2)),
            (false, /, randn(sr(6), 1, 2), 0.66),
            (false, /, randn(sr(7), 2), 5.0 * I),
            (false, /, randn(sr(8), 1), Diagonal(rand(sr(9), 1) .+ 1)),
            (false, /, randn(sr(0), 2, 2), Diagonal(rand(sr(1), 2) .+ 1)),
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
            (false, /, randn(sr(1), 2, size(X, 1)), X)
        end,
        [
            (false, //, [1, 2], 5),
            (false, ==, randn(sr(1), 1, 2), randn(2, 1)),
            (false, Array, randn(sr(2), 1, 2)),
            (false, Bidiagonal, randn(sr(3), 4), randn(sr(4), 3), 'L'),
            (false, CartesianIndices, [1, 2, 3]),
            (false, Diagonal, randn(sr(5), 4)),
            (false, IndexStyle, randn(sr(6), 1, 2)),
            (false, IndexStyle, randn(sr(7), 2, 1), randn(sr(8), 1, 2)),
            (false, LinearIndices, randn(sr(9), 3)),
            (false, LinearIndices, randn(sr(0), 3, 2)),
            (false, PermutedDimsArray, randn(sr(1), 2, 3), [2, 1]),
            (false, SubArray, randn(sr(2), 2, 3), (1:2, 2:3)),
            (false, \, 0.67, randn(sr(3), 2, 2)),
            (false, \, Hermitian(randn(sr(0), 2, 2) + 3I), randn(sr(1), 2)), # missing foreigncall rule
            (false, \, Symmetric(randn(sr(3), 2, 2) + 3I), randn(sr(2), 2)), # missing foreigncall rule
            (false, \, SymTridiagonal(rand(sr(4), 3) .+ 1, randn(sr(5), 2)), randn(sr(6), 3)), # missing foreigncall
            (false, \, 3.0 * I, randn(sr(4), 2)),
            (false, \, 3.0 * I, randn(sr(5), 2, 1)),
            (false, \, UnitLowerTriangular(randn(sr(6), 2, 2)), randn(sr(7), 2)),
            (false, \, UnitUpperTriangular(randn(sr(9), 2, 2)), randn(sr(8), 2)),
            (
                false,
                (A, x) -> cholesky(A)' \ x,
                Symmetric(randn(sr(0), 2, 2) + 5I),
                randn(sr(1), 2),
            ),
            (
                false,
                (A, X) -> cholesky(A)' \ X,
                Symmetric(randn(sr(2), 2, 2) + 5I),
                randn(sr(3), 2, 3),
            ),
            (
                false,
                \,
                adjoint(Bidiagonal(rand(sr(4), 3) .+ 1, randn(sr(5), 2), 'U')),
                randn(sr(6), 3),
            ),
            (
                false,
                \,
                adjoint(Bidiagonal(rand(sr(8), 3) .+ 1, randn(sr(9), 2), 'U')),
                randn(sr(7), 3, 2),
            ),
            (
                false,
                \,
                adjoint(Bidiagonal(rand(sr(9), 3) .+ 1, randn(sr(0), 2), 'L')),
                randn(sr(8), 3),
            ),
            (
                false,
                \,
                adjoint(Bidiagonal(rand(sr(0), 3) .+ 1, randn(sr(1), 2), 'L')),
                randn(sr(9), 3, 2),
            ),
            (false, \, randn(sr(2), 2), randn(sr(3), 2, 2)),
            (false, \, LowerTriangular(rand(sr(4), 2, 2) + 3I), randn(sr(5), 2)),
            (false, \, UpperTriangular(rand(sr(7), 2, 2) + 3I), randn(sr(6), 2)),
            (false, \, Diagonal(rand(sr(9), 2) .+ 1), randn(sr(8), 2)),
            (
                false,
                \,
                Bidiagonal(rand(sr(0), 3) .+ 1, randn(sr(1), 2), 'U'),
                randn(sr(2), 3),
            ),
            (
                false,
                \,
                Bidiagonal(rand(sr(4), 3) .+ 1, randn(sr(5), 2), 'U'),
                randn(sr(6), 3, 2),
            ),
            (
                false,
                \,
                Bidiagonal(rand(sr(7), 3) .+ 1, randn(sr(8), 2), 'L'),
                randn(sr(9), 3),
            ),
            (
                false,
                \,
                Bidiagonal(rand(sr(1), 3) .+ 1, randn(sr(2), 2), 'L'),
                randn(sr(0), 3, 2),
            ),
            (false, \, rand(sr(3), 2, 2) + 3I, randn(sr(4), 2)),
            (false, \, rand(sr(6), 2, 2) + 3I, randn(sr(5), 2, 3)),
            (false, x -> all(>(0), x), randn(sr(7), 2)),
            (false, allunique, randn(sr(8), 2)),
            (false, x -> any(<(0), x), randn(sr(9), 2)),
            (false, append!, randn(sr(1), 3), randn(sr(0), 2)),
            (false, argmax, randn(sr(2), 2)),
            (false, argmin, randn(sr(3), 2)),
            (false, axes, randn(sr(4), 2, 1, 1, 2)),
            (false, circshift, randn(sr(5), 3), 1),
            (false, circshift!, randn(sr(6), 3), randn(sr(7), 3), 1),
            (false, clamp!, randn(sr(8), 3), 0.0, 0.5),
            (false, collect, randn(sr(9), 3)),
            # (false, complex, randn(sr(1), 2)), # Hits non-determinism in https://github.com/JuliaLang/julia/blob/bed2cd540a11544ed4be381d471bbf590f0b745e/base/array.jl#L244
            (false, conj, randn(sr(2), 3)),
            (false, copy!, randn(sr(3), 2), randn(sr(4), 2)),
            (
                false,
                copyto!,
                randn(sr(5), 3),
                CartesianIndices(2:3),
                randn(sr(6), 2),
                CartesianIndices(1:2),
            ),
            (false, copyto!, randn(sr(7), 3), 2, randn(sr(8), 2), 1, 2),
            # (false, copyto!, randn(sr(1), ComplexF64, 3), 2, randn(sr(12), 2), 1, 2), # Hits non-determinism in https://github.com/JuliaLang/julia/blob/bed2cd540a11544ed4be381d471bbf590f0b745e/base/array.jl#L244
            (false, copyto!, randn(sr(1), 3), randn(sr(2), 3)),
            # (false, copyto!, randn(sr(4), ComplexF64, 1, 3), randn(sr(5), 1, 3)), # Hits non-determinism in https://github.com/JuliaLang/julia/blob/bed2cd540a11544ed4be381d471bbf590f0b745e/base/array.jl#L244
            (
                false,
                copyto!,
                PermutedDimsArray(randn(sr(7), 2, 3), [2, 1]),
                randn(sr(6), 3, 2),
            ),
            (false, x -> count(<(0), x), randn(sr(7), 2, 3)),
            (false, (r, A) -> count!(<(0), r, A), randn(sr(8), 2), randn(sr(9), 2, 3)),
            (false, cumprod, randn(sr(0), 3)),
            (
                false,
                (B, A) -> cumprod!(B, A; dims=2),
                randn(sr(1), 2, 2),
                randn(sr(2), 2, 2),
            ),
            (false, cumsum, randn(sr(3), 3)),
            (false, (out, v) -> cumsum!(out, v; dims=1), randn(sr(4), 3), randn(sr(5), 3)),
            (
                false,
                (out, v) -> cumsum!(out, v; dims=2),
                randn(sr(7), 2, 3),
                randn(sr(6), 2, 3),
            ),
            (false, deleteat!, randn(sr(8), 5), 3),
            (false, deleteat!, randn(sr(9), 5), 1),
            (false, deleteat!, randn(sr(0), 5), 5),
            (false, deleteat!, randn(sr(1), 5), 2:3),
            (false, deleteat!, randn(sr(2), 3), [false, true, true]),
            (false, deleteat!, randn(sr(3), 5), [1, 2, 5]),
            (false, diff, randn(sr(4), 5)),
            (false, x -> diff(x; dims=1), randn(sr(5), 3, 2)),
            (false, x -> diff(x; dims=2), randn(sr(6), 3, 2)),
            (false, empty, randn(sr(7), 5)),
            (false, extrema, randn(sr(8), 3)),
            (false, x -> extrema(sin, x), randn(sr(9), 3)),
            (false, x -> extrema(sin, x; dims=1), randn(sr(0), 3, 2)),
            (
                false,
                (r, A) -> extrema!(cos, r, A),
                tuple.(randn(sr(1), 3), randn(sr(2), 3)),
                randn(sr(3), 3, 2),
            ),
            (
                false,
                (r, A) -> extrema!(r, A),
                tuple.(randn(sr(4), 3), randn(sr(5), 3)),
                randn(sr(6), 3, 2),
            ),
            (false, fill!, randn(sr(7), 3), randn(sr(8))),
            (false, fill!, randn(sr(0), 3, 2), randn(sr(9))),
            (false, x -> filter(>(0), x), [0.5, -0.1, -0.4]),
            (false, x -> filter(<(0), x), randn(sr(1), 2, 2)),
            # (false, x -> findall(<(0), x), [0.5, 0.0, -0.3]), # uses invoke, which is not currently supported
            (false, x -> findfirst(<(0), x), [0.5, -0.1, -0.4]),
            (false, x -> findlast(<(0), x), [0.5, -0.1, -0.4]),
            (false, findmax, randn(sr(1), 2, 2)),
            (false, x -> findmax(sin, x), [0.5, -0.1, -0.4]),
            (false, x -> findmax(sin, x), [0.5, -0.1, -0.4]),
            (false, findmin, randn(sr(2), 2, 2)),
            (false, x -> findmin(cos, x), randn(sr(3), 2, 2)),
            (false, first, randn(sr(4), 3)),
            (false, firstindex, randn(sr(5), 3)),
            (false, float, randn(sr(6), 3)),
            (false, (x, i) -> get(() -> 5.0, x, i), randn(sr(7), 3), 2),
            (false, (x, i) -> get(() -> 5.0, x, i), randn(sr(8), 3), 4),
            (false, getindex, randn(sr(9), 5), 1),
            (false, getindex, randn(sr(0), 5), 3),
            (false, getindex, randn(sr(1), 5, 4), 3),
            (false, getindex, randn(sr(2), 5, 4), 2, 3),
            (false, getindex, randn(sr(3), 5), 1:2),
            (false, getindex, randn(sr(4), 5), 2:4),
            (false, getindex, randn(sr(5), 5), :),
            (false, getindex, randn(sr(6), 3, 2), :),
            (false, getindex, randn(sr(7), 5), 1:2:5),
            (false, getindex, randn(sr(8), 5), [1, 3, 5]),
            (false, hash, randn(sr(9), 3, 4), UInt(2)),
            (false, hcat, randn(sr(0), 3, 2)),
            (false, hcat, randn(sr(1), 2, 3), randn(sr(2), 2, 2)),
            (false, imag, randn(sr(3), 2, 3)),
            (false, imag, randn(sr(4), ComplexF64, 2, 3)),
            (false, insert!, randn(sr(5), 3), 2, randn(sr(6))),
            (false, isassigned, randn(sr(7), 5), 3),
            (false, isassigned, randn(sr(8), 5), CartesianIndex(4)),
            (false, isempty, randn(sr(9), 5)),
            (false, isempty, randn(sr(0), 0)),
            (false, isequal, randn(sr(1), 3), randn(sr(2), 3)),
            (false, isreal, randn(sr(3), 3)),
            (false, isreal, randn(sr(4), ComplexF64, 3)),
            (false, iszero, randn(sr(5), 2)),
            (false, iterate, randn(sr(6), 2)),
            (false, iterate, randn(sr(7), 2, 2)),
            (false, keys, randn(sr(8), 2)),
            (false, keytype, randn(sr(9), 2)),
            (false, kron, randn(sr(0)), randn(sr(1), 2, 2)),
            (false, kron, randn(sr(3)), randn(sr(2), 2)),
            (false, kron, randn(sr(4), 2, 2), randn(sr(5))),
            (false, kron, randn(sr(7), 2), randn(sr(6))),
            (false, kron, randn(sr(8), 2), rand(sr(9), 3)),
            (false, kron, randn(sr(1), 2), randn(sr(0), 1, 2)),
            (false, kron, randn(sr(2), 2, 1), randn(sr(3), 3)),
            (false, kron, randn(sr(5), 2, 2), randn(sr(4), 2, 2)),
            (false, kron!, randn(sr(6), 2), randn(sr(7), 2), randn(sr(8))),
            (false, kron!, randn(sr(1), 2, 2), randn(sr(0), 2, 2), randn(sr(9))),
            (false, kron!, randn(sr(2), 2), randn(sr(3)), randn(sr(4), 2)),
            (false, kron!, randn(sr(7), 2, 2), randn(sr(6)), randn(sr(5), 2, 2)),
            (false, kron!, randn(sr(8), 6), randn(sr(9), 2), randn(sr(0), 3)),
            (false, kron!, randn(sr(3), 4, 3), randn(sr(2), 2), randn(sr(1), 2, 3)),
            (false, kron!, randn(sr(4), 4, 6), randn(sr(5), 2, 2), randn(sr(6), 2, 3)),
            (false, kron!, randn(sr(9), 3, 2), randn(sr(8), 1, 2), randn(sr(7), 3)),
            (false, kron!, randn(sr(0), 6, 2), randn(sr(1), 2, 2), randn(sr(3), 3, 1)),
            (false, last, randn(sr(4), 4)),
            (false, lastindex, randn(sr(5), 2)),
            (false, length, randn(sr(6), 2)),
            (false, length, randn(sr(7), 2, 3)),
            (false, map, sin, randn(sr(8), 2)),
            (false, map!, sin, randn(sr(9), 2), randn(sr(0), 2)),
            (false, map!, *, randn(sr(3), 2), randn(sr(2), 2), randn(sr(1), 2)),
            (false, mapreduce, sin, *, randn(sr(4), 2)),
            (false, (f, x) -> mapslices(f, x; dims=1), sum, randn(sr(5), 2, 3)),
            (false, maximum, randn(sr(6), 2)),
            (false, maximum, randn(sr(7), 2, 3)),
            (false, maximum, sin, randn(sr(8), 2)),
            (false, maximum, cos, randn(sr(9), 2, 3)),
            (false, x -> maximum(cos, x; dims=2), randn(sr(0), 3, 2)),
            (false, maximum!, sin, randn(sr(2), 2), randn(sr(1), 2, 3)),
            (false, minimum, randn(sr(3), 2)),
            (false, minimum, randn(sr(4), 2, 3)),
            (false, minimum, sin, randn(sr(5), 2)),
            (false, minimum, cos, randn(sr(6), 2, 3)),
            (false, x -> minimum(cos, x; dims=2), randn(sr(7), 3, 2)),
            (false, minimum!, sin, randn(sr(9), 2), randn(sr(8), 2, 3)),
        ],
        vec(reduce(
            vcat,
            map(product(
                [adjoint(randn(sr(0), 2, 3)), transpose(randn(sr(1), 2, 3))],
                [randn(sr(3), 2), randn(sr(2), 2, 3)],
                [randn(sr(4)), randn(sr(5), 1), randn(sr(6), 3)],
            )) do (A, b, z)
                (false, muladd, A, b, z)
            end,
        )),
        [
            (false, ndims, randn(sr(7), 2)),
            (false, ndims, randn(sr(8), 1, 2, 1, 1, 1)),
            (false, nextind, randn(sr(9), 3, 3), 2),
            (false, nextind, randn(sr(0), 3, 3), CartesianIndex(2, 2)),
            (false, pairs, randn(sr(1), 2, 2)),
            (false, parent, randn(sr(2), 3, 2)),
            (false, parentindices, randn(sr(3), 3, 2)),
            (false, permute!, randn(sr(5), 4), [2, 4, 3, 1]),
            (false, permutedims, randn(sr(6), 3, 2, 1), [2, 1, 3]),
            (false, permutedims, randn(sr(7), 2)),
            (false, permutedims, randn(sr(8), 2, 3)),
            (false, permutedims!, randn(sr(9), 2, 3, 1), randn(sr(0), 3, 2, 1), [2, 1, 3]),
            (false, pop!, randn(sr(1), 5)),
            (false, popat!, randn(sr(2), 5), 1),
            (false, popat!, randn(sr(3), 5), 3),
            (false, popat!, randn(sr(4), 5), 5),
            (false, popat!, randn(sr(5), 5), 7, 3.0),
            (false, popfirst!, randn(sr(6), 5)),
            (false, prepend!, randn(sr(7), 5), randn(sr(8), 3)),
            (false, prevind, randn(sr(9), 2, 3), 5),
            (false, prevind, randn(sr(0), 2, 3), CartesianIndex(2, 2)),
            (false, prod, randn(sr(1), 2)),
            (false, prod, randn(sr(2), 2, 3)),
            (false, x -> prod(x; dims=1), randn(sr(3), 2, 3)),
            (false, x -> prod(sin, x; dims=2), randn(sr(4), 2, 2)),
            (false, prod!, sin, randn(sr(6), 2), randn(sr(5), 2, 3)),
            (false, prod!, randn(sr(7), 2), randn(sr(8), 2, 3)),
            (false, promote_shape, randn(sr(0), 2), randn(sr(9), 2)),
            (false, promote_shape, randn(sr(1), 2), randn(sr(2), 2, 1, 1)),
            (false, push!, randn(sr(3), 5), 4.0),
            (false, pushfirst!, randn(sr(4), 5), 3.0),
            (false, real, randn(sr(5), 3, 2, 1)),
            (false, reduce, *, randn(sr(6), 2)),
            (false, reduce, *, randn(sr(7), 3, 2)),
            (false, repeat, randn(sr(8), 2), 2, 3),
            (false, (x, i, o) -> repeat(x; inner=i, outer=o), randn(sr(9), 2), 2, 3),
            (false, reshape, randn(sr(0), 3, 2), 2, 3),
            (false, reshape, randn(sr(1), 3, 2), :, 3),
            (false, reshape, randn(sr(2), 3, 2), (6,)),
            (false, x -> (resize!(x, 10); x[6:end] .= x[1:5]), randn(sr(3), 5)),
            (false, reverse, randn(sr(4), 3)),
            (false, reverse!, randn(sr(5), 3)),
            (false, reverseind, randn(sr(6), 5), 3),
            (false, selectdim, randn(sr(7), 3), 1, 2),
            # (false, setdiff!, randn(sr(8), 3), randn(sr(9), 2)), # weird control flow problem
            (false, setindex!, randn(sr(0), 3), randn(sr(1)), 1),
            (false, setindex!, randn(sr(2), 3), randn(sr(3), 2), 1:2),
            (false, setindex!, randn(sr(4), 3), randn(sr(5)), CartesianIndex(2)),
            (false, setindex!, randn(sr(7), 3, 2), randn(sr(6)), 5),
            (false, setindex!, randn(sr(8), 3, 2), randn(sr(9)), 2, 2),
            (false, setindex!, randn(sr(0), 3, 2), randn(sr(1)), CartesianIndex(2, 1)),
            (false, setindex!, randn(sr(2), 3, 2), randn(sr(3), 2, 2), 2:3, 1:2),
            (false, setindex!, randn(sr(5), 3, 2), randn(sr(4), 2, 2), 1:2, :),
            (false, setindex!, randn(sr(6), 3, 2), randn(sr(7)), CartesianIndex(2), 1),
            (true, similar, randn(sr(8), 3, 2)),
            (false, size, randn(sr(9), 2)),
            (false, size, randn(sr(0), 3, 2)),
            (false, size, randn(sr(1), 3, 2, 1)),
            (false, sizeof, randn(sr(2), 5)),
            (false, sort, randn(sr(3), 3)),
            (false, sort!, randn(sr(4), 3)),
            (false, sortperm, randn(sr(0), 3)),
            (false, sortperm!, [1, 2, 3], randn(sr(1), 3)),
            (false, x -> sortslices(x; dims=1), randn(sr(2), 2, 3)),
            (false, splice!, randn(sr(5), 5), 1:2),
            (false, splice!, randn(sr(6), 5), 1:0),
            (false, splice!, randn(sr(7), 5), 1:5),
            (false, splice!, randn(sr(8), 5), 1:2, randn(sr(6), 3)),
            (false, splice!, randn(sr(9), 5), 3:2, randn(sr(5), 4)),
            (false, splice!, randn(sr(0), 5), 1),
            (false, splice!, randn(sr(1), 5), 1, randn(sr(3))),
            (false, splice!, randn(sr(2), 5), 1, randn(sr(4), 3)),
            (false, stride, randn(sr(7), 3, 2), 1),
            (false, sum, randn(sr(8), 2, 3)),
            (false, x -> sum(x; dims=1), randn(sr(9), 3, 2)),
            (false, (f, x) -> sum(f, x; dims=2), sin, randn(sr(0), 3, 2)),
            (false, sum!, randn(sr(2), 1, 3), randn(sr(1), 2, 3)),
            (false, sum!, sin, randn(sr(3), 1, 3), randn(sr(4), 2, 3)),
            (false, transpose, randn(sr(5), 3)),
            (false, transpose, randn(sr(6), 3, 2)),
            (false, transpose, randn(sr(7), 1, 3)),
            # (false, unique, randn(sr(8), 3)), # hits invoke, which is not currently supported
            (false, valtype, randn(sr(9), 3)),
            (false, vcat, randn(sr(0), 2)),
            (false, vcat, randn(sr(1), 2, 2)),
            (false, vcat, randn(sr(2), 3), randn(2)),
            (false, vcat, randn(sr(3), 3, 2), randn(sr(4), 2, 2)),
            (false, vec, randn(sr(5), 2)),
            (false, vec, randn(sr(6), 2, 3)),
            (false, view, randn(sr(7), 2, 3), 1:2, 1:2),
            (false, view, randn(sr(8), 3, 2), 1:3),
            (false, view, randn(sr(9), 3, 2), :, 1),
            (false, view, randn(sr(0), 3, 2), :, :),
            (false, zero, randn(sr(1), 3)),
            (false, zero, randn(sr(2), 2, 3)),
        ]
    )
        rng = StableRNG(123456)
        @info map(Core.Typeof, (f, x...))
        sig = Tuple{Core.Typeof(f), map(Core.Typeof, x)...}
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig; interp)
        @test in_f(deepcopy(x)...) == f(deepcopy(x)...)
        # val, _ = Taped.trace(f, x...; ctx=Taped.RMC())
        # test_taped_rrule!!(rng, f, deepcopy(x)...; interface_only, perf_flag=:none)
    end
end
