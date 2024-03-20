using Turing

using ReverseDiff
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
    μ ~ truncated(Normal(1, 2), 0.1, 10)
    σ ~ truncated(Normal(1, 2), 0.1, 10)
    
    x .~ LogNormal(μ, σ)   
end

# LDA example -- copied over from
# https://github.com/TuringLang/Turing.jl/issues/668#issuecomment-1153124051
function _make_data(D, K, V, N, α, η)
    β = Matrix{Float64}(undef, V, K)
    for k in 1:K
        β[:,k] .= rand(Dirichlet(η))
    end

    θ = Matrix{Float64}(undef, K, D)
    z = Vector{Int}(undef, D * N)
    w = Vector{Int}(undef, D * N)
    doc = Vector{Int}(undef, D * N)
    i = 0
    for d in 1:D
        θ[:,d] .= rand(Dirichlet(α))
        for n in 1:N
            i += 1
            z[i] = rand(Categorical(θ[:, d]))
            w[i] = rand(Categorical(β[:, z[i]]))
            doc[i] = d
        end
    end
    return (D=D, K=K, V=V, N=N, α=α, η=η, z=z, w=w, doc=doc, θ=θ, β=β)
end

data = let D = 2, K = 2, V = 160, N = 290
    _make_data(D, K, V, N, ones(K), ones(V))
end

# LDA with vectorization and manual log-density accumulation
@model function LatentDirichletAllocationVectorizedCollapsedMannual(
    D, K, V, α, η, w, doc
)
    β ~ filldist(Dirichlet(η), K)
    θ ~ filldist(Dirichlet(α), D)

    log_product = log.(β * θ)
    Turing.@addlogprob! sum(log_product[CartesianIndex.(w, doc)])
    # Above is equivalent to below
    #product = β * θ
    #dist = [Categorical(product[:,i]) for i in 1:D]
    #w ~ arraydist([dist[doc[i]] for i in 1:length(doc)])
end

function build_turing_problem(rng, model, example=nothing)
    ctx = Turing.DefaultContext()
    vi = example === nothing ? Turing.SimpleVarInfo(model) : Turing.SimpleVarInfo(example)
    vi_linked = Turing.link(vi, model)
    ldp = Turing.LogDensityFunction(vi_linked, model, ctx)
    test_function = Base.Fix1(Turing.LogDensityProblems.logdensity, ldp)
    d = Turing.LogDensityProblems.dimension(ldp)
    return test_function, randn(rng, d)
end

@testset "turing" begin
    interp = Taped.TInterp()
    @testset "$(typeof(model))" for (interface_only, name, model, ex) in vcat(
        Any[
            (false, "simple_model", simple_model(), nothing),
            (false, "demo", demo(), nothing),
            (
                false,
                "broadcast_demo",
                broadcast_demo(rand(LogNormal(1.5, 0.5), 1_000)),
                nothing,
            ),
            # (
            #     false,
            #     "CollapsedLDA",
            #     LatentDirichletAllocationVectorizedCollapsedMannual(
            #         data.D, data.K, data.V, data.α, data.η, data.w, data.doc,
            #     ),
            # ), doesn't currently work with SimpleVarInfo
        ],
        Any[
            (false, "demo_$n", m, Turing.DynamicPPL.TestUtils.rand_prior_true(m)) for
                (n, m) in enumerate(Turing.DynamicPPL.TestUtils.DEMO_MODELS[1:11])
        ],
    )
        @info name
        rng = sr(123)
        f, x = build_turing_problem(rng, model, ex)
        TestUtils.test_derived_rule(
            sr(123456), f, x;
            perf_flag=:none, interface_only=true, is_primitive=false, interp
        )

        # rule = build_rrule(interp, _typeof((f, x)))
        # interp_rule, in_f = TestUtils.set_up_gradient_problem(f, x);
        # interp_codualed_args = map(zero_codual, (in_f, f, x));
        # codualed_args = map(zero_codual, (f, x))
        # TestUtils.value_and_gradient!!(rule, codualed_args...)

        # # @profview run_many_times(1_000, TestUtils.value_and_gradient!!, rule, codualed_args...)

        # primal = @benchmark $f($x)
        # # interpreted = @benchmark $in_f($f, $x)
        # interp_gradient = @benchmark(TestUtils.value_and_gradient!!($interp_rule, $interp_codualed_args...))
        # gradient = @benchmark(TestUtils.value_and_gradient!!($rule, $codualed_args...))

        # println("primal")
        # display(primal)
        # println()

        # println("interpreted gradient")
        # display(interp_gradient)
        # println()

        # println("gradient")
        # display(gradient)
        # println()

        # @show time(interp_gradient) / time(primal)
        # @show time(gradient) / time(primal)

        # try
        #     tape = ReverseDiff.GradientTape(f, x);
        #     ReverseDiff.gradient!(tape, x);
        #     result = zeros(size(x));
        #     ReverseDiff.gradient!(result, tape, x)

        #     revdiff = @benchmark ReverseDiff.gradient!($result, $tape, $x)
        #     println("ReverseDiff")
        #     display(revdiff)
        #     println()
        #     @show time(revdiff) / time(primal)
        # catch
        #     display("revdiff failed")
        # end

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
