using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake
using Mooncake: Mooncake
using Mooncake.TestUtils
using Mooncake.TestUtils: test_rule
using DynamicExpressions
using StableRNGs: StableRNG
using DifferentiationInterface: AutoMooncake, gradient, prepare_gradient

using Test

@testset "Basic usage checks" begin
    let
        # Build up expression
        operators = OperatorEnum(1 => (cos, sin), 2 => (+, -, *, /))
        x1, x2 = (Expression(Node{Float64}(; feature=i); operators) for i in 1:2)

        f = x1 + cos(x2 - 0.2) + 0.5
        X = randn(StableRNG(0), 3, 100)

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
end

@testset "Gradient of tree parameters" begin
    let
        operators = OperatorEnum(1 => (cos, sin), 2 => (+, -, *, /))
        x1 = Expression(Node{Float64}(; feature=1); operators)

        #  simple closed‑form ground truth: ∂/∂c sum(x1 + c) = N
        N = 100
        X = randn(StableRNG(0), 3, N)
        expr = x1 + 0.0      # constant in the tree

        eval_sum = let X = X
            f -> sum(f(X))
        end

        backend = AutoMooncake(; config=nothing)
        prep = prepare_gradient(eval_sum, backend, expr)
        dexpr = gradient(eval_sum, prep, backend, expr)

        const_tangent = dexpr.fields.tree.children[2].x.val
        @test const_tangent ≈ N
    end
end

@testset "TestUtils systematic tests" begin
    let
        operators = OperatorEnum(
            1 => (cos, sin, exp, log, abs), 2 => (+, -, *, /), 3 => (fma, max)
        )

        x1 = Expression(Node{Float64,3}(; feature=1); operators)
        x2 = Expression(Node{Float64,3}(; feature=2); operators)

        # Various expression types - using only operators that exist
        expressions = [
            x1,
            x1 + 1.0,
            cos(x1),
            x1 * x2 + sin(x1 - 0.5),
            fma(x1, x2, x2 + x2),
            fma(max(x1, x2, 2.0 * x1), x2 * 2.1, x1 * 3.2),
        ]

        # Test cases for expression evaluation
        X = randn(StableRNG(0), 3, 20)

        make_eval_sum(expr) = X -> sum(expr(X))
        test_cases = [
            (;
                interface_only = false,
                perf_flag = :none,
                is_primitive = false,
                unsafe_perturb = true,
                fargs = (make_eval_sum(expr), X),
                label = string(expr),
            )
            for expr in expressions
        ]

        @testset "$(test_case.label)" for test_case in test_cases
            test_rule(
                StableRNG(1),
                test_case.fargs...;
                test_case.interface_only,
                test_case.perf_flag,
                test_case.is_primitive,
                test_case.unsafe_perturb,
            )
        end
    end
end
