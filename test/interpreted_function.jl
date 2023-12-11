@testset "interpreted_function" begin

    # Check that inlining doesn't / does happen as expected.
    @testset "TapedInterpreter" begin
        @testset "non-primitive continues to be inlined away" begin

            # A non-primitive is present in the IR for contains_non_primitive. It is
            # inlined away under usual interpretation, and should also be inlined away
            # when doing AD.
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
            @test length(ad_ir.stmts) == 2
            @test ad_ir.stmts.inst[1].head == :invoke
            @test ad_ir.stmts.inst[1].args[2] == GlobalRef(Taped, :sin)
            @test ad_ir.stmts.inst[2] isa Core.ReturnNode
        end
        @testset "primitive is no longer inlined away" begin

            # A primitive is present in the IR for contains_primitive. It is inlined away
            # under usual interpretation, but should not be when doing AD.
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
            @test length(ad_ir.stmts) == 2
            @test ad_ir.stmts.inst[1].head == :invoke
            @test ad_ir.stmts.inst[1].args[2] == GlobalRef(Taped, :a_primitive)
            @test ad_ir.stmts.inst[2] isa Core.ReturnNode
        end
        @testset "deep primitive is not inlined away" begin

            # A non-primitive is immediately visible in the IR, but this non-primitive is
            # usually inlined away to reveal a primitive. This primitive is _also_ usually
            # inlined away, but should not be when doing AD. This case is not handled if
            # various bits of information are not properly propagated in the compiler.
            sig = Tuple{typeof(Taped.contains_primitive_behind_call), Float64}

            # Pre-condition: both functions should be inlined away under usual conditions.
            usual_ir = Base.code_ircode_by_type(sig)[1][1]
            @assert length(usual_ir.stmts) == 2
            @assert usual_ir.stmts.inst[1].head == :invoke
            @assert usual_ir.stmts.inst[1].args[2] == GlobalRef(Taped, :sin)
            @assert usual_ir.stmts.inst[2] isa Core.ReturnNode

            # Should not inline away under AD compilation.
            interp = Taped.TapedInterpreter(Taped.DefaultCtx())
            ad_ir = Base.code_ircode_by_type(sig; interp)[1][1]
            @test length(ad_ir.stmts) == 2
            @test ad_ir.stmts.inst[1].head == :invoke
            @test ad_ir.stmts.inst[1].args[2] == GlobalRef(Taped, :a_primitive)
            @test ad_ir.stmts.inst[2] isa Core.ReturnNode
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

    interp = Taped.TInterp()
    Taped.flush_interpreted_function_cache!()

    # nothings inserted for consistency with generate_test_functions.
    @testset "$f, $(map(Core.Typeof, x))" for (a, b, f, x...) in vcat(
        Any[
            (nothing, nothing, Taped.foo, 5.0),
            (nothing, nothing, Taped.bar, 5.0, 4.0),
            (nothing, nothing, identity, 5.0),
            (nothing, nothing, Taped.const_tester),
            (nothing, nothing, Taped.type_unstable_argument_eval, sin, 5.0),
            (nothing, nothing, Taped.pi_node_tester, Ref{Any}(5.0)),
            (nothing, nothing, Taped.pi_node_tester, Ref{Any}(5)),
            (nothing, nothing, Taped.intrinsic_tester, 5.0),
            (nothing, nothing, Taped.goto_tester, 5.0),
            (nothing, nothing, Taped.new_tester, 5.0, :hello),
            (nothing, nothing, Taped.new_2_tester, 4.0),
            (nothing, nothing, Taped.type_unstable_tester, Ref{Any}(5.0)),
            (nothing, nothing, Taped.type_unstable_tester_2, Ref{Real}(5.0)),
            (nothing, nothing, Taped.type_unstable_function_eval, Ref{Any}(sin), 5.0),
            (nothing, nothing, Taped.phi_const_bool_tester, 5.0),
            (nothing, nothing, Taped.phi_const_bool_tester, -5.0),
            (nothing, nothing, Taped.avoid_throwing_path_tester, 5.0),
            (nothing, nothing, Taped.simple_foreigncall_tester, randn(5)),
            (nothing, nothing, Taped.foreigncall_tester, randn(5)),
            (nothing, nothing, Taped.no_primitive_inlining_tester, 5.0),
            (nothing, nothing, Taped.varargs_tester, 5.0),
            (nothing, nothing, Taped.varargs_tester, 5.0, 4),
            (nothing, nothing, Taped.varargs_tester, 5.0, 4, 3.0),
            (nothing, nothing, Taped.varargs_tester_2, 5.0),
            (nothing, nothing, Taped.varargs_tester_2, 5.0, 4),
            (nothing, nothing, Taped.varargs_tester_2, 5.0, 4, 3.0),
            (nothing, nothing, Taped.varargs_tester_3, 5.0),
            (nothing, nothing, Taped.varargs_tester_3, 5.0, 4),
            (nothing, nothing, Taped.varargs_tester_3, 5.0, 4, 3.0),
            (nothing, nothing, Taped.varargs_tester_4, 5.0),
            (nothing, nothing, Taped.varargs_tester_4, 5.0, 4),
            (nothing, nothing, Taped.varargs_tester_4, 5.0, 4, 3.0),
            (nothing, nothing, Taped.splatting_tester, 5.0),
            (nothing, nothing, Taped.splatting_tester, (5.0, 4.0)),
            (nothing, nothing, Taped.splatting_tester, (5.0, 4.0, 3.0)),
            # (nothing, nothing, Taped.unstable_splatting_tester, Ref{Any}(5.0)),
            # (nothing, nothing, Taped.unstable_splatting_tester, Ref{Any}((5.0, 4.0))),
            # (nothing, nothing, Taped.unstable_splatting_tester, Ref{Any}((5.0, 4.0, 3.0))),
            (
                nothing,
                nothing,
                LinearAlgebra._modify!,
                LinearAlgebra.MulAddMul(5.0, 4.0),
                5.0,
                randn(5, 4),
                (5, 4),
            ), # for Bool comma,
            (
                nothing, nothing,
                mul!, transpose(randn(3, 5)), randn(5, 5), randn(5, 3), 4.0, 3.0,
            ), # static_parameter,
            (nothing, nothing, Xoshiro, 123456),
        ],
        TestResources.generate_test_functions(),
    )
        @info "$f, $x"
        sig = Tuple{Core.Typeof(f), map(Core.Typeof, x)...}
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig; interp)

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
        # original = @benchmark $(Ref(f))[]($(Ref(x))[]...)
        # r = @benchmark $(Ref(in_f))[]($(Ref(x))[]...)

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
