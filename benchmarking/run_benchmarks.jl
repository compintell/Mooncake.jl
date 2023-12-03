using Pkg
Pkg.develop(path=joinpath(@__DIR__, ".."))

using BenchmarkTools, Random, Taped, Test

using Taped: CoDual, generate_hand_written_rrule!!_test_cases

function generate_rrule!!_benchmarks(rng::AbstractRNG, args)

    # Generate CoDuals etc.
    primals = map(x -> x isa CoDual ? primal(x) : x, args)
    dargs = map(x -> x isa CoDual ? tangent(x) : randn_tangent(rng, x), args)
    codual_args = map(CoDual, primals, dargs)
    suite = BenchmarkGroup()

    # Benchmark primal.
    suite["primal"] = @benchmarkable(
        (a[1])((a[2:end])...);
        setup=(a = deepcopy($primals)),
        evals=1,    
    )

    # Benchmark forwards-pass.
    suite["forwards"] = @benchmarkable(
        Taped.rrule!!(ca...);
        setup=(ca = deepcopy($(codual_args))),
        evals=1,
    )

    # Benchmark pullback.
    suite["pullback"] = @benchmarkable(
        x[2]((tangent(x[1])), map(tangent, ca)...),
        setup=(ca = deepcopy($codual_args); x = Taped.rrule!!(ca...)),
        evals=1,
    )

    return suite
end

function benchmark_rrules!!(rng::AbstractRNG)

    # Benchmark the performance of all benchmarks.
    test_case_data = [
        generate_hand_written_rrule!!_test_cases(Val(:avoiding_non_differentiable_code)),
        generate_hand_written_rrule!!_test_cases(Val(:blas)),
        generate_hand_written_rrule!!_test_cases(Val(:builtins)),
        generate_hand_written_rrule!!_test_cases(Val(:foreigncall)),
        generate_hand_written_rrule!!_test_cases(Val(:iddict)),
        generate_hand_written_rrule!!_test_cases(Val(:lapack)),
        generate_hand_written_rrule!!_test_cases(Val(:low_level_maths)),
        generate_hand_written_rrule!!_test_cases(Val(:misc)),
        generate_hand_written_rrule!!_test_cases(Val(:umlaut_internals_rules)),
        generate_hand_written_rrule!!_test_cases(Val(:unrolled_function)),
    ]
    test_cases = reduce(vcat, map(first, test_case_data))
    memory = map(last, test_case_data)

    GC.@preserve memory begin
        results = map(enumerate(test_cases)) do (n, x)
            args = (x[4:end]..., )
            @info "$n / $(length(test_cases))", args
            suite = generate_rrule!!_benchmarks(rng, args)
            return (x, BenchmarkTools.run(suite; verbose=true, seconds=3))
        end
    end

    # Compute performance ratio for all cases.
    ratios = map(results) do result
        result_dict = result[2]
        primal_time = time(minimum(result_dict["primal"]))
        forwards_time = time(minimum(result_dict["forwards"]))
        pullback_time = time(minimum(result_dict["pullback"]))
        _range = result[1][3]
        return (
            tag=result[1],
            forwards_range=_range === nothing ? default_hand_written_ratios() : _range,
            pullback_range=_range === nothing ? default_hand_written_ratios() : _range,
            primal_time=primal_time,
            forwards_time=forwards_time,
            pullback_time=pullback_time,
            forwards_ratio=forwards_time / primal_time,
            pullback_ratio=pullback_time / primal_time,
        )
    end
    return ratios
end

default_hand_written_ratios() = (lb=1e-3, ub=10.0)

between(x, (lb, ub)) = lb < x && x < ub

function flag_concerning_performance(ratios)
    @testset "detect concerning performance" begin
        @testset for ratio in ratios
            @test between(ratio.forwards_ratio, ratio.forwards_range)
            @test between(ratio.pullback_ratio, ratio.pullback_range)
        end
    end
end

function main()
    ratios = benchmark_rrules!!(Xoshiro(123456))
    flag_concerning_performance(ratios)
    return nothing
end

main()
