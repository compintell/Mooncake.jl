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
        vi = Turing.SimpleVarInfo(model)
        vi_linked = Turing.link(vi, model)
        ldp = Turing.LogDensityFunction(vi_linked, model, ctx)
        return Turing.LogDensityProblems.logdensity(ldp, x)
    end

    m = model_function(args...)
    v = Turing.SimpleVarInfo(m)
    v_linked = Turing.link(v, m)
    _ldp = Turing.LogDensityFunction(v_linked, m, Turing.DefaultContext())
    d = Turing.LogDensityProblems.dimension(_ldp)

    return test_function, randn(rng, d)
end

@testset "turing" begin
    interp = Taped.TInterp()
    @testset "$model_function" for (interface_only, model_function, args...) in [
        (false, simple_model),
        (false, demo),
    ]
        @info model_function
        rng = sr(123)
        f, x = build_turing_problem(rng, model_function, args...)

        x = (x, model_function, args...)

        @show f(x...)
        sig = Tuple{Core.Typeof(f), map(Core.Typeof, x)...}
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp)
        if interface_only
            in_f(f, deepcopy(x)...)
        else
            @test has_equal_data(in_f(f, deepcopy(x)...), f(deepcopy(x)...))
        end
        display(@benchmark $f($x...))
        println()
        display(@benchmark $in_f($f, $x...))
        println()
    end
end
