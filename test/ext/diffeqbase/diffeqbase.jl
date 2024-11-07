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

function build_and_solve(u0, tspan, p, sensealg)
    prob = ODEProblem(lotka_volterra!, u0, (0.0, 1.0), p)
    sol = DiffEqBase.solve(prob, Tsit5(); abstol=1e-14, reltol=1e-14, sensealg)
    return sum(sum(sol.u[end]))
end

@testset "diffeqbase" begin
    u0 = [1.0, 1.0]
    tspan = (0.0, 1.0)
    p = [1.5, 1.0, 4.0, 1.0]

    @testset "$sensealg" for sensealg in Any[
        # ForwardSensitivity(), # falls over
        ForwardDiffSensitivity(),
        BacksolveAdjoint(),
        GaussAdjoint(),
        InterpolatingAdjoint(),
        QuadratureAdjoint(),
        # ReverseDiffAdjoint() # falls over -- some kind of stackoverflow
        # TrackerAdjoint(), # stackoverflow
        # ZygoteAdjoint() # falls over
        # ForwardLSS(), # falls over -- might be user error on my part
        # AdjointLSS(), # falls over
        # NILSS(10, 10), # falls over
        # NILSAS(10, 10, 15), # falls over
    ]
        Mooncake.TestUtils.test_rule(
            Xoshiro(123), build_and_solve, u0, tspan, p, sensealg;
            is_primitive=false, interface_only=false, debug_mode=false,
        )
    end

    # display(@benchmark build_and_solve($u0, $tspan, $p))

    # rule = Mooncake.build_rrule(build_and_solve, u0, tspan, p);
    # display(@benchmark Mooncake.value_and_gradient!!($rule, build_and_solve, $u0, $tspan, $p))
end
