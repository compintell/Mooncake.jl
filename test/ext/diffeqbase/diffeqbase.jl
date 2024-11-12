using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DiffEqBase, Mooncake, OrdinaryDiffEqTsit5, Random, SciMLSensitivity, Test
using DiffEqBase: solve

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
const prob = ODEProblem(lotka_volterra!, u0, (0.0, 1.0), p)

function build_and_solve(u0, tspan, p, sensealg)
    _prob = remake(prob; u0, p)
    sol = solve(_prob, Tsit5(); abstol=1e-14, reltol=1e-14, sensealg, saveat=0.01)
    return sum(sol) + sum(sum(sol.u[end])) + sum(sol[end])
end

function matrix_ode!(du, u, p, t)
    du .= reshape(p, 4, 4) * u
    return nothing
end

u0_mat = rand(4, 8)
p_mat = rand(16)

# Have to use remake with a const global in order to get good performance.
const matrix_prob = ODEProblem(matrix_ode!, u0_mat, (0.0, 1.0), p_mat)

function build_and_solve_mat(u0, tspan, p, sensealg)
    _prob = remake(matrix_prob; u0, p)
    sol = solve(_prob, Tsit5(); abstol=1e-14, reltol=1e-14, sensealg, saveat=0.01)
    return sol[:, :, end]
end

@testset "diffeqbase" begin
    vjps = [false, true, EnzymeVJP(), ZygoteVJP(), ReverseDiffVJP(), ReverseDiffVJP(true)]
    reduced_vjps = [false, EnzymeVJP(), ReverseDiffVJP(), ReverseDiffVJP(true)]

    # These cases are excluded because Zygote also does not successfully work on them.
    excluded_cases = Any[
        BacksolveAdjoint(; autojacvec=false),
        BacksolveAdjoint(; autojacvec=true),
        QuadratureAdjoint(; autojacvec=EnzymeVJP()),
    ]

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
            is_primitive=false, debug_mode=false,
        )

        sensealg in excluded_cases && continue
        Mooncake.TestUtils.test_rule(
            Xoshiro(123), build_and_solve_mat, u0_mat, tspan, p_mat, sensealg;
            is_primitive=false, debug_mode=false,
        )
    end
end
