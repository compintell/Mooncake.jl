

# REFACTOR NEW IMPLEMENTATION TO AVOID HAVING AN EXTRA SPECIAL CASE -- IT SHOULD BE
# POSSIBLE TO DO IT IN A WAY THAT IS PERFORMANT NOW THAT I UNDERSTAND WHAT IS GOING ON! 
@testset "interpreted_function" begin
    # nothings inserted for consistency with generate_test_functions.
    @testset "$f" for (a, b, f, x...) in vcat(
        Any[
            (nothing, nothing, Taped.foo, 5.0),
            (nothing, nothing, Taped.bar, 5.0, 4.0),
            (nothing, nothing, identity, 5.0),
            (nothing, nothing, Taped.const_tester, ),
            (nothing, nothing, Taped.intrinsic_tester, 5.0),
            (nothing, nothing, Taped.goto_tester, 5.0),
            # (nothing, nothing, Taped.new_tester, 5.0, :hello),
            # (nothing, nothing, Taped.new_2_tester, 4.0),
            # (nothing, nothing, Taped.type_unstable_tester, Ref{Any}(5.0)),
            # (nothing, nothing, Taped.pi_node_tester, Ref{Any}(5.0)),
            # (nothing, nothing, Taped.avoid_throwing_path_tester, 5.0),
            # (nothing, nothing, Taped.foreigncall_tester, 5.0),
        ],
        # TestResources.generate_test_functions(),
    )
        @info "$f, $(typeof(x))"
        sig = Tuple{typeof(f), map(typeof, x)...}
        in_f = Taped.InterpretedFunction(sig)

        # Check primal.
        @test in_f(x...) == f(x...)
        @test in_f(x...) == f(x...) # run twice to check for non-determinism.

        # Check gradient.
        TestUtils.test_rrule!!(
            Xoshiro(123456), in_f, x...;
            perf_flag=:none, interface_only=false, is_primitive=false,
        )

        # test_taped_rrule!!(rng, f, deepcopy(x)...; interface_only=false, perf_flag=:none)

        # Only bother to check performance if the original programme does not allocate.
        original = @benchmark $f($x...)
        r = @benchmark $in_f($x...)

        __rrule!! = Taped.build_rrule!!(in_f)
        codual_x = map(zero_codual, x)
        rrule_timing = @benchmark($__rrule!!(zero_codual($in_f), $codual_x...))
        out, pb!! = __rrule!!(zero_codual(in_f), codual_x...)
        df = zero_codual(in_f)
        overall_timing = @benchmark Taped.to_benchmark($__rrule!!, $df, $codual_x)
        println("original")
        display(original)
        println()
        println("taped")
        display(r)
        println()
        println("rrule")
        display(rrule_timing)
        println()
        println("overall")
        display(overall_timing)
        println()

        if allocs(original) == 0
            @test allocs(r) == 0
        end
    end
end
