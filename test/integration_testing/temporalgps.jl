using AbstractGPs, KernelFunctions, TemporalGPs

build_gp() = to_sde(GP(Matern12Kernel()), SArrayStorage(Float64))

temporalgps_logpdf_tester(x, y, s) = logpdf(build_gp()(x, s), y)

@testset "temporalgps" begin
    x = range(-5.0; step=0.1, length=10_000)
    s = 1.0
    y = rand(build_gp()(x, s))

    f = temporalgps_logpdf_tester
    x = (x, y, s)
    sig = _typeof((temporalgps_logpdf_tester, x...))
    @info "$sig"
    interp = Tapir.PInterp()
    TestUtils.test_derived_rule(
        Xoshiro(123456), f, x...;
        interp, perf_flag=:none, interface_only=false, is_primitive=false
    )

    # codual_args = map(zero_codual, (f, x...))
    # rule = Tapir.build_rrule(interp, sig)

    # primal = @benchmark $f($x...)
    # gradient = @benchmark(TestUtils.to_benchmark($rule, $codual_args...))

    # println("primal")
    # display(primal)
    # println()

    # println("gradient")
    # display(gradient)
    # println()

    # @show time(gradient) / time(primal)

    # # @profview run_many_times(100, f, x...)
    # TestUtils.to_benchmark(rule, codual_args...)
    # @profview run_many_times(10, TestUtils.to_benchmark, rule, codual_args...)
    # Profile.clear()
    # @profile run_many_times(10, TestUtils.to_benchmark, rule, codual_args...)
    # pprof()
end
