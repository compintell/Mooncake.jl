using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake
using Mooncake: Mooncake
using Mooncake.TestUtils
using DynamicExpressions, Random
using DifferentiationInterface: AutoMooncake, gradient, prepare_gradient

using Test

@testset "Basic usage checks" begin
    # Build up expression
    operators = OperatorEnum(1 => (cos, sin), 2 => (+, -, *, /))
    x1, x2 = (Expression(Node{Float64}(; feature=i); operators) for i in 1:2)

    f = x1 + cos(x2 - 0.2) + 0.5
    X = randn(MersenneTwister(0), 3, 100)

    eval_sum = let f = f
        X -> sum(f(X))
    end
    backend = AutoMooncake(; config=nothing)
    prep = prepare_gradient(eval_sum, backend, X)
    dX = gradient(eval_sum, prep, backend, X)

    # analytic derivative: df/dx1 = 1, df/dx2 = -sin(x2 - 0.2), df/dx3 = 0
    dX_ref = zeros(size(X))
    dX_ref[1, :] .= 1
    dX_ref[2, :] .= -sin.(X[2, :] .- 0.2)
    # third row already zero
    @test isapprox(dX, dX_ref; rtol=1e-10, atol=0)
end

@testset "Gradient of tree parameters" begin
    operators = OperatorEnum(1 => (cos, sin), 2 => (+, -, *, /))
    x1 = Expression(Node{Float64}(; feature=1); operators)

    #  simple closed‑form ground truth: ∂/∂c sum(x1 + c) = N
    N = 100
    Xc = randn(MersenneTwister(0), 3, N)
    expr = x1 + 0.0      # constant in the tree

    eval_sum_c = let X = Xc
        f -> sum(f(X))
    end

    backend = AutoMooncake(; config=nothing)
    prep = prepare_gradient(eval_sum_c, backend, expr)
    dexpr = gradient(eval_sum_c, prep, backend, expr)

    const_tangent = dexpr.fields.tree.children[2].x.val
    @test const_tangent ≈ N
end
