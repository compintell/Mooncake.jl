using Turing

# using CSV, DataFrames, ReverseDiff
# turing_bench_results = DataFrame(
#     :name => String[],
#     :primal => [],
#     :interp => [],
#     :gradient => [],
#     :reversediff => [],
# )

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

@model broadcast_demo(x) = begin
    μ ~ TruncatedNormal(1, 2, 0.1, 10)
    σ ~ TruncatedNormal(1, 2, 0.1, 10)
    
    x .~ LogNormal(μ, σ)   
end

function build_turing_problem(rng, model)
    ctx = Turing.DefaultContext()
    vi = Turing.SimpleVarInfo(model)
    vi_linked = Turing.link(vi, model)
    ldp = Turing.LogDensityFunction(vi_linked, model, ctx)
    test_function = Base.Fix1(Turing.LogDensityProblems.logdensity, ldp)
    d = Turing.LogDensityProblems.dimension(ldp)
    return test_function, randn(rng, d)
end


@testset "turing" begin
    interp = Taped.TInterp()
    @testset "$(typeof(model))" for (interface_only, name, model) in vcat(
        Any[
            (false, "simple_model", simple_model()),
            (false, "demo", demo()),
            (false, "broadcast_demo", broadcast_demo(rand(LogNormal(1.5, 0.5), 1_000))),
        ],
    )
        @info typeof(model)
        rng = sr(123)
        f, x = build_turing_problem(rng, model)

        sig = Tuple{Core.Typeof(f), Core.Typeof(x)}
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp);
        if interface_only
            in_f(f, deepcopy(x))
        else
            @show in_f(f, deepcopy(x))
            @show f(deepcopy(x))
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

        # rule, in_f = TestUtils.set_up_gradient_problem(f, x);
        # codualed_args = map(zero_codual, (in_f, f, x));
        # TestUtils.value_and_gradient!!(rule, codualed_args...)

        # # @profview run_many_times(1_000, TestUtils.value_and_gradient!!, rule, codualed_args...)

        # primal = @benchmark $f($x)
        # interpreted = @benchmark $in_f($f, $x)
        # gradient = @benchmark(TestUtils.value_and_gradient!!($rule, $codualed_args...))
        # revdiff = @benchmark ReverseDiff.gradient!($result, $tape, $x)

        # println("primal")
        # display(primal)
        # println()

        # println("interpreted")
        # display(interpreted)
        # println()

        # println("gradient")
        # display(gradient)
        # println()

        # println("revdiff")
        # display(revdiff)
        # println()
    
        # push!(turing_bench_results, (name, primal, interpreted, gradient, revdiff))
    end
end

# function process_turing_bench_results(df::DataFrame)
#     out_df = DataFrame(
#         :name => df.name,
#         :primal => map(time, df.primal),
#         :interp => map(time, df.interp),
#         :gradient => map(time, df.gradient),
#         :reversediff => map(time, df.reversediff),
#     )
#     CSV.write("turing_benchmarks.csv", out_df)
# end
# process_turing_bench_results(turing_bench_results)
