
@testset "interpreted_function" begin
    # nothings inserted for consistency with generate_test_functions.
    @testset "$f" for (a, b, f, x...) in vcat(
        Any[
            (nothing, nothing, Taped.foo, 5.0),
            (nothing, nothing, Taped.bar, 5.0, 4.0),
            (nothing, nothing, Taped.const_tester, ),
            (nothing, nothing, Taped.intrinsic_tester, 5.0),
            (nothing, nothing, Taped.goto_tester, 5.0),
            (nothing, nothing, Taped.new_tester, 5.0),
            (nothing, nothing, Taped.type_unstable_tester, Ref{Any}(5.0)),
            (nothing, nothing, Taped.pi_node_tester, Ref{Any}(5.0)),
            (nothing, nothing, Taped.avoid_throwing_path_tester, 5.0),
        ],
        # TestResources.generate_test_functions(),
    )
        @info "$f, $(typeof(x))"
        sig = Tuple{typeof(f), map(typeof, x)...}
        # display(Base.code_ircode_by_type(sig)[1][1])
        # println()
        in_f = Taped.InterpretedFunction(sig)
        @test in_f(x...) == f(x...)

        # # Only bother to check performance if the original programme does not allocate.
        # original = @benchmark $f($x...)
        # if allocs(original) == 0
        #     r = @benchmark $in_f($x...)
        #     println("original")
        #     display(original)
        #     println()
        #     println("taped")
        #     display(r)
        #     println()

        #     @test allocs(r) == 0
        # end
    end
end
