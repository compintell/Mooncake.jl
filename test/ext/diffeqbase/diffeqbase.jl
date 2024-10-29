using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DiffEqBase, Mooncake, OrdinaryDiffEqTsit5, Random, SciMLSensitivity, Test

# function Mooncake.tangent_type(::Type{<:DiffEqBase.FunctionWrappersWrappers.FunctionWrappersWrapper})
#     return NoTangent
# end

# Mooncake.tangent_type(::Type{<:ODEFunction}) = NoTangent

# Mooncake.tangent_type(::Type{<:ODEProblem}) = NoTangent

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

function build_and_solve(u0, tspan, p)
    prob = ODEProblem(lotka_volterra!, u0, (0.0, 1.0), p)
    return sum(sum(DiffEqBase.solve(prob, Tsit5(); abstol=1e-14, reltol=1e-14).u[end]))
end

@testset "diffeqbase" begin
    u0 = [1.0, 1.0]
    tspan = (0.0, 1.0)
    p = [1.5, 1.0, 4.0, 1.0]
    Mooncake.TestUtils.test_rule(
        Xoshiro(123), build_and_solve, u0, tspan, p;
        is_primitive=false, interface_only=false, debug_mode=true,
    )
end

# foo(prob) = sum(sum(DiffEqBase.solve(prob, Tsit5()).u))
# @benchmark foo($prob)

# rule = Mooncake.build_rrule(foo, prob);
# @benchmark Mooncake.value_and_gradient!!($rule, $foo, $prob)
