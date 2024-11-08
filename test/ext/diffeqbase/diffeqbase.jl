using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DiffEqBase, Mooncake, OrdinaryDiffEqTsit5, Random, SciMLSensitivity, Test

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

u0 = [1.0, 1.0]
tspan = (0.0, 1.0)
p = [1.5, 1.0, 4.0, 1.0]

# Have to use remake with a const global in order to get good performance.
const __prob = ODEProblem(lotka_volterra!, u0, (0.0, 1.0), p)

function build_and_solve(u0, tspan, p, sensealg)
    _prob = remake(__prob; u0, p)
    sol = DiffEqBase.solve(_prob, Tsit5(); abstol=1e-14, reltol=1e-14, sensealg)
    return sum(sum(sol.u[end]))
end

@testset "diffeqbase" begin
    vjps = [false, true, EnzymeVJP(), ZygoteVJP(), ReverseDiffVJP(), ReverseDiffVJP(true)]
    reduced_vjps = [false, EnzymeVJP(), ReverseDiffVJP(), ReverseDiffVJP(true)]
    @testset "$sensealg" for sensealg in vcat(
        [ForwardDiffSensitivity()],
        [BacksolveAdjoint(; autojacvec=vjp) for vjp in vjps],
        [GaussAdjoint(; autojacvec=vjp) for vjp in reduced_vjps],
        [InterpolatingAdjoint(; autojacvec=vjp) for vjp in vjps],
        [QuadratureAdjoint(; autojacvec=vjp) for vjp in reduced_vjps],
    )
        @info sensealg
        Mooncake.TestUtils.test_rule(
            Xoshiro(123), build_and_solve, u0, tspan, p, sensealg;
            is_primitive=false, interface_only=false, debug_mode=false,
        )
    end
end
