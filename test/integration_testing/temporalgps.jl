using AbstractGPs, KernelFunctions, TemporalGPs

build_gp(k) = to_sde(GP(k), SArrayStorage(Float64))

temporalgps_logpdf_tester(k, x, y, s) = logpdf(build_gp(k)(x, s), y)

@testset "temporalgps" begin

    xs = Any[
        collect(range(-5.0; step=0.1, length=1_000)),
        RegularSpacing(0.0, 0.1, 1_000),
    ]
    base_kernels = Any[Matern12Kernel(), Matern32Kernel()]
    kernels = vcat(base_kernels, [with_lengthscale(k, 1.1) for k in base_kernels])
    @testset "$k, $(typeof(x))" for (k, x) in vec(collect(Iterators.product(kernels, xs)))
        s = 1.0
        y = rand(build_gp(k)(x, s))
        f = temporalgps_logpdf_tester
        sig = _typeof((temporalgps_logpdf_tester, k, x, y, s))
        @info "$sig"
        interp = Tapir.PInterp()
        TestUtils.test_derived_rule(
            Xoshiro(123456), f, k, x, y, s;
            interp, perf_flag=:none, interface_only=false, is_primitive=false
        )
    end


    # x = range(-5.0; step=0.1, length=10_000)


    # f = temporalgps_logpdf_tester
    # x = (x, y, s)
    # sig = _typeof((temporalgps_logpdf_tester, x...))
    # @info "$sig"
    # interp = Tapir.PInterp()
    # TestUtils.test_derived_rule(
    #     Xoshiro(123456), f, x...;
    #     interp, perf_flag=:none, interface_only=false, is_primitive=false
    # )

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
