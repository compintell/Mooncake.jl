using Turing

@model function simple_model()
    y ~ Normal()
end

@model function demo()
    # Assumptions
    σ2 ~ LogNormal() # tweaked from InverseGamma due to control flow issues.
    σ = sqrt(σ2 + 1e-3)
    μ ~ Normal(0.0, σ)
  
    # Observations
    x ~ Normal(μ, σ)
    y ~ Normal(μ, σ)
end

function build_turing_problem(rng, model_function, args...)

    function test_function(x, model_function, args...)
        model = model_function(args...)
        ctx = Turing.DefaultContext()
        vi = Turing.VarInfo(model)
        vi_linked = Turing.link(vi, model)
        ldp = Turing.LogDensityFunction(vi_linked, model, ctx)
        return Turing.LogDensityProblems.logdensity(ldp, x)
    end

    m = model_function(args...)
    v = Turing.VarInfo(m)
    v_linked = Turing.link(v, m)
    _ldp = Turing.LogDensityFunction(v_linked, m, Turing.DefaultContext())
    d = Turing.LogDensityProblems.dimension(_ldp)

    return test_function, randn(rng, d)
end

@testset "turing" begin
    @testset "$model_function" for (interface_only, model_function, args...) in [
        (false, simple_model),
        (false, demo),
    ]
        rng = sr(123)
        f, x = build_turing_problem(rng, model_function, args...)
        display(Taped.trace(f, x, model_function, args...; ctx=Taped.RMC()))
        println()
        TestUtils.test_taped_rrule!!(rng, f, x, model_function, args...)
    end
end
