using Pkg
Pkg.develop(path=joinpath(@__DIR__, ".."))

using
    AbstractGPs,
    BenchmarkTools,
    CSV,
    DataFrames,
    Enzyme,
    KernelFunctions,
    LinearAlgebra,
    Plots,
    PrettyTables,
    Random,
    ReverseDiff,
    Taped,
    Test,
    Zygote

using Taped:
    CoDual,
    generate_hand_written_rrule!!_test_cases,
    generate_derived_rrule!!_test_cases,
    InterpretedFunction,
    TestUtils,
    TInterp,
    _typeof

using Taped.TestUtils: _deepcopy, to_benchmark, set_up_gradient_problem

function zygote_to_benchmark(ctx, x::Vararg{Any, N}) where {N}
    out, pb = Zygote._pullback(ctx, x...)
    return pb(out)
end

function rd_to_benchmark!(result, tape, x)
    return ReverseDiff.gradient!(result, tape, x)
end

should_run_benchmark(args...) = true

# Test out the performance of a hand-written sum function, so we can be confident that there
# is no rule. Note that ReverseDiff has a (seemingly not fantastic) hand-written rule for
# sum.
function _sum(x::AbstractArray{<:Real})
    y = 0.0
    n = 0
    while n < length(x)
        n += 1
        y += x[n]
    end
    return y
end

# Zygote has rules for both sum and kron, so it's interesting to compare this against the
# other frameworks because they don't have rules for kron, and maybe for not for sum.
_kron_sum(x::AbstractMatrix{<:Real}, y::AbstractMatrix{<:Real}) = sum(kron(x, y))

# Zygote (should) use the ChainRules projection functionality to handle the more interesting
# types surrounding the arrays.
function _kron_view_sum(x::AbstractMatrix{<:Real}, y::AbstractMatrix{<:Real})
    return _kron_sum(view(x, 1:20, 1:20), UpperTriangular(y))
end

# No one has a rule for this.
function _naive_map_sin_cos_exp(x::AbstractArray{<:Real})
    y = similar(x)
    for n in eachindex(x)
        y[n] = sin(cos(exp(x[n])))
    end
    return sum(y)
end

should_run_benchmark(::Val{:zygote}, ::typeof(_naive_map_sin_cos_exp), x) = false

# RD and Zygote have a rule for this.
_map_sin_cos_exp(x::AbstractArray{<:Real}) = sum(map(x -> sin(cos(exp(x))), x))

# Only Zygote has a rule for this.
_broadcast_sin_cos_exp(x::AbstractArray{<:Real}) = sum(sin.(cos.(exp.(x))))

# Different frameworks have rules for this to differing degrees. Zygote has rules for just
# about all of the operations.
_simple_mlp(W2, W1, Y, X) = sum(abs2, Y - W2 * map(x -> x * (0 <= x), W1 * X))

# Only Zygote and Taped can actually handle this. Note that Taped only has rules for BLAS
# and LAPACK stuff, not explicit rules for things like the squared euclidean distance.
# Consequently, Zygote is at a major advantage.
_gp_lml(x, y, s) = logpdf(GP(SEKernel())(x, s), y)

should_run_benchmark(::Val{:reverse_diff}, ::typeof(_gp_lml), x...) = false
should_run_benchmark(::Val{:enzyme}, ::typeof(_gp_lml), x...) = false

function _generate_gp_inputs()
    x = collect(range(0.0; step=0.2, length=128))
    s = 1.0
    y = rand(GP(SEKernel())(x, s))
    return x, y, s
end

"""
    generate_inter_framework_tests()

Constructs a set of benchmarks which can be used to compare between AD frameworks.
Outputs a vector of tuples. Each tuples comprises a function (first element) and arguments
at which its value and pullback should be computed (remaining elements).

Arguments must comprise only scalars and arrays, and output must be either a scalar or
an array.
"""
function generate_inter_framework_tests()
    return Any[
        (sum, randn(100)),
        (_sum, randn(100)),
        # (_kron_sum, randn(20, 20), randn(40, 40)),
        # (_kron_view_sum, randn(40, 30), randn(40, 40)),
        # (_naive_map_sin_cos_exp, randn(10, 10)),
        # (_map_sin_cos_exp, randn(10, 10)),
        # (_broadcast_sin_cos_exp, randn(10, 10)),
        # (_simple_mlp, randn(128, 256), randn(256, 128), randn(128, 70), randn(128, 70)),
        # (_gp_lml, _generate_gp_inputs()...),
    ]
end

function benchmark_rules!!(
    test_case_data,
    default_ratios,
    include_other_frameworks::Bool,
    tune_benchmarks::Bool,
)
    test_cases = reduce(vcat, map(first, test_case_data))
    memory = map(x -> x[2], test_case_data)
    ranges = reduce(vcat, map(x -> x[3], test_case_data))
    GC.@preserve memory begin
        results = map(enumerate(test_cases)) do (n, args)
            @info "$n / $(length(test_cases))", _typeof(args)
            suite = BenchmarkGroup()

            # Benchmark primal.
            primals = map(x -> x isa CoDual ? primal(x) : x, args)
            suite["primal"] = @benchmarkable(
                (a[1][])((a[2][])...);
                setup=(a = (Ref($primals[1]), Ref(_deepcopy($primals[2:end])))),
            )

            # Benchmark AD via Taped.
            rule, in_f = set_up_gradient_problem(args...)
            coduals = map(x -> x isa CoDual ? x : zero_codual(x), args)
            suite["taped"] = @benchmarkable(
                to_benchmark($rule, zero_codual($in_f), $coduals...);
            )

            if include_other_frameworks

                if should_run_benchmark(Val(:zygote), args...)
                    suite["zygote"] = @benchmarkable(
                        zygote_to_benchmark($(Zygote.Context()), $primals...)
                    )
                end

                if should_run_benchmark(Val(:reverse_diff), args...)
                    tape = ReverseDiff.GradientTape(primals[1], primals[2:end])
                    compiled_tape = ReverseDiff.compile(tape)
                    result = map(x -> randn(size(x)), primals[2:end])
                    suite["rd"] = @benchmarkable(
                        rd_to_benchmark!($result, $compiled_tape, $primals[2:end])
                    )
                end

                if should_run_benchmark(Val(:enzyme), args...)
                    dup_args = map(x -> Duplicated(x, randn(size(x))), primals[2:end])
                    suite["enzyme"] = @benchmarkable(
                        autodiff(Reverse, $primals[1], Active, $dup_args...)
                    )
                end
            end

            if tune_benchmarks
                @info "tuning"
                tune!(suite)
            end
            @info "running"
            return (args, run(suite; verbose=true))
        end
    end
    return combine_results.(results, ranges, Ref(default_ratios))
end

function combine_results(result, _range, default_range)
    d = result[2]
    primal_time = time(minimum(d["primal"]))
    taped_time = time(minimum(d["taped"]))
    zygote_time = in("zygote", keys(d)) ? time(minimum(d["zygote"])) : missing
    rd_time = in("rd", keys(d)) ? time(minimum(d["rd"])) : missing
    ez_time = in("enzyme", keys(d)) ? time(minimum(d["enzyme"])) : missing
    return (
        tag=string((result[1][1], map(Taped._typeof, result[1][2:end])...)),
        primal_time=primal_time,
        taped_time=taped_time,
        taped_ratio=taped_time / primal_time,
        zygote_time=zygote_time,
        zygote_ratio=zygote_time / primal_time,
        rd_time=rd_time,
        rd_ratio=rd_time / primal_time,
        enzyme_time=ez_time,
        enzyme_ratio=ez_time / primal_time,
        range=_range === nothing ? default_range : _range,
    )
end

function benchmark_hand_written_rrules!!(rng_ctor)
    test_case_data = map([
        :avoiding_non_differentiable_code,
        :blas,
        :builtins,
        :foreigncall,
        :iddict,
        :lapack,
        :low_level_maths,
        :misc,
        :new,
    ]) do s
        test_cases, memory = generate_hand_written_rrule!!_test_cases(rng_ctor, Val(s))
        ranges = map(x -> x[3], test_cases)
        return map(x -> x[4:end], test_cases), memory, ranges
    end
    return benchmark_rules!!(test_case_data, (lb=1e-3, ub=25.0), false, false)
end

function benchmark_derived_rrules!!(rng_ctor)
    test_case_data = map([
        :test_utils
    ]) do s
        test_cases, memory = generate_derived_rrule!!_test_cases(rng_ctor, Val(s))
        ranges = map(x -> x[3], test_cases)
        return map(x -> x[4:end], test_cases), memory, ranges
    end
    return benchmark_rules!!(test_case_data, (lb=1e-3, ub=150), false, false)
end

function benchmark_inter_framework_rules()
    test_cases = generate_inter_framework_tests()
    memory = []
    ranges = fill(nothing, length(test_cases))
    return benchmark_rules!!([(test_cases, memory, ranges)], (lb=0.1, ub=150), true, true)
end

function flag_concerning_performance(ratios)
    @testset "detect concerning performance" begin
        @testset for ratio in ratios
            @test ratio.range.lb < ratio.taped_ratio < ratio.range.ub
        end
    end
end

"""
    plot_ratio_histogram!(df::DataFrame)

Constructs a histogram of the `taped_ratio` field of `df`, with formatting that is
well-suited to the numbers typically found in this field.
"""
function plot_ratio_histogram!(df::DataFrame)
    bin = 10.0 .^ (0.0:0.05:6.0)
    xlim = extrema(bin)
    histogram(df.taped_ratio; xscale=:log10, xlim, bin, title="log", label="")
end

function create_inter_ad_benchmarks()
    results = benchmark_inter_framework_rules()
    df = DataFrame(results)[:, [:tag, :taped_ratio, :zygote_ratio, :rd_ratio, :enzyme_ratio]]

    tag_map = Dict{String, String}(
        "(sum, Vector{Float64})" => "sum",
        "(_sum, Vector{Float64})" => "_sum",
        "(_kron_sum, Matrix{Float64}, Matrix{Float64})" => "kron",
        "(_kron_view_sum, Matrix{Float64}, Matrix{Float64})" => "kron types",
        "(_naive_map_sin_cos_exp, Matrix{Float64})" => "naive map",
        "(_map_sin_cos_exp, Matrix{Float64})" => "map",
        "(_broadcast_sin_cos_exp, Matrix{Float64})" => "broadcast",
        "(_simple_mlp, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64})" => "mlp",
        "(_gp_lml, Vector{Float64}, Vector{Float64}, Float64)" => "gp",
    )

    df.label = map(t -> tag_map[t], df.tag)

    tool_map = Dict{Symbol, String}(
        :taped_ratio => "Taped",
        :zygote_ratio => "Zygote",
        :rd_ratio => "ReverseDiff",
        :enzyme_ratio => "Enzyme",
    )

    plt = plot(
        yscale=:log10,
        legend=:topright,
        title="Ratio of AD Time to Primal Time (Log Scale)",
    )
    for key in keys(tool_map)
        plot!(plt, df.label, df[:, key]; label=tool_map[key], marker=:circle, xrotation=45)
    end
    Plots.savefig(plt, "bench/benchmark_results.png")

    df_formatted = DataFrame(
        label = df.label,
        taped = string.(round.(df.taped_ratio; sigdigits=3)),
        zygote = string.(round.(df.zygote_ratio; sigdigits=3)),
        reverse_diff = string.(round.(df.rd_ratio; sigdigits=3)),
        enzyme = string.(round.(df.enzyme_ratio; sigdigits=3)),
    )
    # CSV.write("bench/benchmarking_results.csv", df_formatted)

    # Write table to text file.
    open("bench/benchmark_results.txt"; write=true) do io
        pretty_table(io, df_formatted)
    end
end

function main()
    perf_group = get(ENV, "PERF_GROUP", "hand_written")
    @info perf_group
    println(perf_group)
    if perf_group == "hand_written"
        flag_concerning_performance(benchmark_hand_written_rrules!!(Xoshiro))
    elseif perf_group == "derived"
        flag_concerning_performance(benchmark_derived_rrules!!(Xoshiro))
    elseif perf_group == "comparison"
        create_inter_ad_benchmarks()
    else
        throw(error("perf_group=$(perf_group) is not recognised"))
    end
end
