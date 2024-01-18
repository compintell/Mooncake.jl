using Turing

# using ReverseDiff

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

function build_turing_problem(rng, model)
    ctx = Turing.DefaultContext()
    vi = Turing.SimpleVarInfo(model)
    vi_linked = Turing.link(vi, model)
    ldp = Turing.LogDensityFunction(vi_linked, model, ctx)
    test_function = Base.Fix1(Turing.LogDensityProblems.logdensity, ldp)

    m = model
    v = Turing.SimpleVarInfo(m)
    v_linked = Turing.link(v, m)
    _ldp = Turing.LogDensityFunction(v_linked, m, Turing.DefaultContext())
    d = Turing.LogDensityProblems.dimension(_ldp)

    return test_function, randn(rng, d)
end

@testset "turing" begin
    interp = Taped.TInterp()
    @testset "$model" for (interface_only, model) in vcat(
        Any[
            (false, simple_model()),
            (false, demo()),
        ],
    )
        @info model
        rng = sr(123)
        f, x = build_turing_problem(rng, model)

        sig = Tuple{Core.Typeof(f), Core.Typeof(x)}
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp);
        if interface_only
            in_f(f, deepcopy(x))
        else
            @test has_equal_data(in_f(f, deepcopy(x)), f(deepcopy(x)))
        end

        TestUtils.test_rrule!!(
            sr(123456), in_f, f, x;
            perf_flag=:none, interface_only=true, is_primitive=false,
        )

        # tape = ReverseDiff.GradientTape(f, x);
        # ReverseDiff.gradient!(tape, x);
        # result = zeros(size(x));
        # ReverseDiff.gradient!(result, tape, x)

        # __rrule = Taped.build_rrule!!(in_f);
        # codualed_args = map(zero_codual, (in_f, f, x));
        # TestUtils.gradient(__rrule, codualed_args[1], codualed_args[2:end])[end]

        # # @profview run_many_times(10, TestUtils.gradient, __rrule, codualed_args[1], codualed_args[2:end])

        # println("primal")
        # display(@benchmark $f($x))
        # println()
        # println("interpreted")
        # display(@benchmark $in_f($f, $x))
        # println()
        # println("gradient")
        # display(@benchmark TestUtils.gradient($__rrule, $codualed_args[1], $(codualed_args[2:end])))
        # println()
        # println("gradient (ReverseDiff.jl)")
        # display(@benchmark ReverseDiff.gradient!($result, $tape, $x))
        # println()
    end
end
