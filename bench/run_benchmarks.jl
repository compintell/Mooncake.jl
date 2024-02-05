using Pkg
Pkg.develop(path=joinpath(@__DIR__, ".."))

using BenchmarkTools, CSV, DataFrames, Plots, Random, Taped, Test

using Taped:
    CoDual,
    generate_hand_written_rrule!!_test_cases,
    generate_derived_rrule!!_test_cases,
    InterpretedFunction,
    TestUtils,
    TInterp

using Taped.TestUtils: _deepcopy, to_benchmark

function benchmark_rules!!(test_case_data, default_ratios)
    test_cases = reduce(vcat, map(first, test_case_data))
    memory = map(x -> x[2], test_case_data)
    ranges = reduce(vcat, map(x -> x[3], test_case_data))
    GC.@preserve memory begin
        results = map(enumerate(test_cases)) do (n, args)
            @info "$n / $(length(test_cases))", Core.Typeof(args)
            suite = BenchmarkGroup()

            # Benchmark primal.
            primals = map(x -> x isa CoDual ? primal(x) : x, args)
            suite["primal"] = @benchmarkable(
                (a[1])((a[2:end])...);
                setup=(a = ($primals[1], _deepcopy($primals[2:end])...)),
            )

            # Benchmark pullback.
            rule, in_f = TestUtils.set_up_gradient_problem(args...)
            coduals = map(x -> x isa CoDual ? x : zero_codual(x), args)
            suite["value_and_pb"] = @benchmarkable(
                to_benchmark($rule, zero_codual($in_f), $coduals...);
            )

            return (args, BenchmarkTools.run(suite, verbose=true, seconds=3))
        end
    end
    return combine_results.(results, ranges, Ref(default_ratios))
end

function combine_results(result, _range, default_range)
    result_dict = result[2]
    primal_time = time(minimum(result_dict["primal"]))
    value_and_pb_time = time(minimum(result_dict["value_and_pb"]))
    return (
        tag=string(Core.Typeof((result[1]..., ))),
        primal_time=primal_time,
        value_and_pb_time=value_and_pb_time,
        value_and_pb_ratio=value_and_pb_time / primal_time,
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
    return benchmark_rules!!(test_case_data, (lb=1e-3, ub=25.0))
end

function benchmark_derived_rrules!!(rng_ctor)
    # Only testing a subset of the cases because there are still problems in the subset of
    # the cases.
    test_case_data = map([
        # :avoiding_non_differentiable_code,
        # :blas,
        # :builtins,
        # :foreigncall,
        # :iddict,
        # :lapack,
        # :low_level_maths,
        # :misc,
        # :new,
        :test_utils
    ]) do s
        test_cases, memory = generate_derived_rrule!!_test_cases(rng_ctor, Val(s))
        ranges = map(x -> x[3], test_cases)
        return map(x -> x[4:end], test_cases), memory, ranges
    end
    return benchmark_rules!!(test_case_data, (lb=0.1, ub=150))
end

function flag_concerning_performance(ratios)
    between(x, (lb, ub)) = lb < x && x < ub
    @testset "detect concerning performance" begin
        @testset for ratio in ratios
            @test between(ratio.value_and_pb_ratio, ratio.range)
        end
    end
end

"""
    plot_ratio_histogram!(df::DataFrame)

Computes the fields fields `gradient_time` and `gradient_ratio` to `df`. The former is
simply `forwards_time + pullbacktime`, which the latter is `gradient_time / primal_time`.

Then displays the `gradient_ratio` field as a pair of histograms, one with a linear x-scale,
and one with a log x-scale.
"""
function plot_ratio_histogram!(df::DataFrame)
    bin = 10.0 .^ (0.0:0.05:6.0)
    xlim = extrema(bin)
    histogram(df.value_and_pb_ratio; xscale=:log10, xlim, bin, title="log", label="")
end

function main()
    perf_group = get(ENV, "PERF_GROUP", "hand_written")
    if perf_group == "hand_written"
        flag_concerning_performance(benchmark_hand_written_rrules!!(Xoshiro))
    elseif perf_group == "derived"
        flag_concerning_performance(benchmark_derived_rrules!!(Xoshiro))
    else
        throw(error("perf_group=$(perf_group) is not recognised"))
    end
end
