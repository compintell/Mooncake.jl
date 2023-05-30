relu(x::Array{Float64}) = max.(0.0, x)

Taped.isprimitive(::Taped.IC, ::typeof(relu), x::Array{Float64}) = true

function Taped.InplaceData(::typeof(relu), x::Array{Float64})
    return InplaceData{typeof(relu)}((y=similar(x), ))
end

function (data::InplaceData{typeof(relu)})(x::Array{Float64})
    data.data.y .= max.(0.0, x)
    return data.data.y
end

mlp(x::Array{Float64}, A::Matrix{Float64}, B::Matrix{Float64}) = B * relu(A * x)

@testset "inplace" begin
    N = 1_000
    x = randn(5N, 100)
    A = randn(6N, 5N)
    B = randn(3N, 6N)
    _, tape = Taped.trace(mlp, x, A, B; ctx=Taped.IC())
    inplace_tape = Taped.to_inplace(tape)
    y = play!(inplace_tape, mlp, x, A, B)
    @test y ≈ mlp(x, A, B)

    # If you look at these benchmarks, you'll see that the total memory allocated is
    # orders of magnitude less with the in-place-ified version.
    # using BenchmarkTools
    # display(@benchmark play!($inplace_tape, mlp, $x, $A, $B))
    # println()
    # display(@benchmark mlp($x, $A, $B))
    # println()
end
