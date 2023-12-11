@testset "interpreted_function" begin

    # Check that inlining doesn't / does happen as expected.
    @testset "TapedInterpreter" begin

        @testset "primitive is no longer inlined away" begin
            sig = Tuple{typeof(Taped.contains_primitive), Float64}

            # Pre-condition: must inline away under usual compilation.
            usual_ir = Base.code_ircode_by_type(sig)[1][1]
            @assert length(usual_ir.stmts) == 2
            @assert usual_ir.stmts.inst[1].head == :invoke
            @assert usual_ir.stmts.inst[1].args[2] == GlobalRef(Taped, :sin)
            @assert usual_ir.stmts.inst[2] isa Core.ReturnNode

            # Should not inline away under AD compilation.
            interp = Taped.TapedInterpreter(Taped.DefaultCtx())
            ad_ir = Base.code_ircode_by_type(sig; interp)[1][1]
            @test length(ad_ir.stmts) == 2 &&
                ad_ir.stmts.inst[1].head == :invoke &&
                ad_ir.stmts.inst[1].args[2] == GlobalRef(Taped, :a_primitive) &&
                ad_ir.stmts.inst[2] isa Core.ReturnNode
        end
        @testset "non-primitive continues to be inlined away" begin
            sig = Tuple{typeof(Taped.contains_non_primitive), Float64}

            # Pre-condition: must inline away under usual compilation.
            usual_ir = Base.code_ircode_by_type(sig)[1][1]
            @assert length(usual_ir.stmts) == 2
            @assert usual_ir.stmts.inst[1].head == :invoke
            @assert usual_ir.stmts.inst[1].args[2] == GlobalRef(Taped, :sin)
            @assert usual_ir.stmts.inst[2] isa Core.ReturnNode

            # Should continue to inline away under AD compilation.
            interp = Taped.TapedInterpreter(Taped.DefaultCtx())
            ad_ir = Base.code_ircode_by_type(sig; interp)[1][1]
            @test length(ad_ir.stmts) == 2 &&
                ad_ir.stmts.inst[1].head == :invoke &&
                ad_ir.stmts.inst[1].args[2] == GlobalRef(Taped, :sin) &&
                ad_ir.stmts.inst[2] isa Core.ReturnNode
        end
    end

    # Check correctness and performance of the ArgInfo type. We really need everything to
    # infer correctly.
    @testset "ArgInfo: $Tx, $(x), $is_va" for (Tx, x, is_va) in Any[

        # No varargs examples.
        Any[Tuple{Nothing, Float64}, (5.0,), false],
        Any[Tuple{Nothing, Float64, Int}, (5.0, 3), false],
        Any[Tuple{Nothing, Type{Float64}}, (Float64, ), false],
        Any[Tuple{Nothing, Type{Any}}, (Any, ), false],

        # Varargs examples.
        Any[Tuple{Nothing, Tuple{Float64}}, (5.0, ), true],
        Any[Tuple{Nothing, Tuple{Float64, Int}}, (5.0, 3), true],
        Any[Tuple{Nothing, Float64, Tuple{Int}}, (5.0, 3), true],
        Any[Tuple{Nothing, Float64, Tuple{Int, Float64}}, (5.0, 3, 4.0), true],
    ]
        ai = Taped.arginfo_from_argtypes(Tx, is_va)
        @test @inferred Taped.load_args!(ai, x) === nothing
    end

    # nothings inserted for consistency with generate_test_functions.
    @testset "$f" for (a, b, f, x...) in vcat(
        Any[
            Any[nothing, nothing, Taped.foo, 5.0],
            Any[nothing, nothing, Taped.bar, 5.0, 4.0],
            Any[nothing, nothing, identity, 5.0],
            Any[nothing, nothing, Taped.const_tester],
            Any[nothing, nothing, Taped.intrinsic_tester, 5.0],
            Any[nothing, nothing, Taped.goto_tester, 5.0],
            Any[nothing, nothing, Taped.new_tester, 5.0, :hello],
            Any[nothing, nothing, Taped.new_2_tester, 4.0],
            Any[nothing, nothing, Taped.type_unstable_tester, Ref{Any}(5.0)],
            Any[nothing, nothing, Taped.type_unstable_tester_2, Ref{Real}(5.0)],
            # Any[nothing, nothing, Taped.type_unstable_function_eval, Ref{Any}(sin), 5.0],
            Any[nothing, nothing, Taped.phi_const_bool_tester, 5.0],
            Any[nothing, nothing, Taped.phi_const_bool_tester, -5.0],
            Any[nothing, nothing, Taped.pi_node_tester, Ref{Any}(5.0)],
            Any[nothing, nothing, Taped.avoid_throwing_path_tester, 5.0],
            Any[nothing, nothing, Taped.foreigncall_tester, randn(5)],
            Any[nothing, nothing, Taped.no_primitive_inlining_tester, 5.0],
            Any[nothing, nothing, Taped.varargs_tester, 5.0],
            Any[nothing, nothing, Taped.varargs_tester, 5.0, 4],
            Any[nothing, nothing, Taped.varargs_tester, 5.0, 4, 3.0],
            Any[nothing, nothing, Taped.varargs_tester_2, 5.0],
            Any[nothing, nothing, Taped.varargs_tester_2, 5.0, 4],
            Any[nothing, nothing, Taped.varargs_tester_2, 5.0, 4, 3.0],
            Any[nothing, nothing, Taped.varargs_tester_3, 5.0],
            Any[nothing, nothing, Taped.varargs_tester_3, 5.0, 4],
            Any[nothing, nothing, Taped.varargs_tester_3, 5.0, 4, 3.0],
            Any[nothing, nothing, Taped.varargs_tester_4, 5.0],
            Any[nothing, nothing, Taped.varargs_tester_4, 5.0, 4],
            Any[nothing, nothing, Taped.varargs_tester_4, 5.0, 4, 3.0],
        ],
        TestResources.generate_test_functions(),
    )
        @info "$f, $x"
        sig = Tuple{Core.Typeof(f), map(Core.Typeof, x)...}
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig)

        # Verify correctness.
        @assert f(x...) == f(x...) # primal runs
        @test in_f(x...) == f(x...)
        @test in_f(x...) == f(x...) # run twice to check for non-determinism.
        TestUtils.test_rrule!!(
            Xoshiro(123456), in_f, x...;
            perf_flag=:none, interface_only=false, is_primitive=false,
        )

        # rng = Xoshiro(123456)
        # test_taped_rrule!!(rng, f, deepcopy(x)...; interface_only=false, perf_flag=:none)

        # # Only bother to check performance if the original programme does not allocate.
        # original = @benchmark $f($x...)
        # r = @benchmark $in_f($x...)

        # __rrule!! = Taped.build_rrule!!(in_f)
        # codual_x = map(zero_codual, x)
        # rrule_timing = @benchmark($__rrule!!(zero_codual($in_f), $codual_x...))
        # out, pb!! = __rrule!!(zero_codual(in_f), codual_x...)
        # df = zero_codual(in_f)
        # overall_timing = @benchmark Taped.to_benchmark($__rrule!!, $df, $codual_x)
        # println("original")
        # display(original)
        # println()
        # println("taped")
        # display(r)
        # println()
        # println("rrule")
        # display(rrule_timing)
        # println()
        # println("overall")
        # display(overall_timing)
        # println()

        # if allocs(original) == 0
        #     @test allocs(r) == 0
        # end
    end
end
