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
            @show sig
            display(usual_ir)
            println()
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

    @testset "TypedGlobalRef" begin
        @testset "tracks changes" begin
            global __x_for_gref = 5.0
            r = TypedGlobalRef(GlobalRef(Main, :__x_for_gref))
            @test r[] == 5.0
            global __x_for_gref = 4.0
            @test r[] == 4.0
        end
        @testset "is type stable" begin
            global __y_for_gref::Float64 = 5.0
            r = TypedGlobalRef(GlobalRef(Main, :__y_for_gref))
            @test @inferred(r[]) == 5.0
            global __y_for_gref = 4.0
            @test @inferred(r[]) == 4.0
        end
    end

    # Check correctness and performance of the ArgInfo type. We really need everything to
    # infer correctly.
    @testset "ArgInfo: $Tx, $(x), $is_va" for (Tx, x, is_va) in Any[

        # No varargs examples.
        Any[Tuple{Float64}, (5.0,), false],
        Any[Tuple{Float64, Int}, (5.0, 3), false],
        Any[Tuple{Type{Float64}}, (Float64, ), false],
        Any[Tuple{Type{Any}}, (Any, ), false],

        # Varargs examples.
        Any[Tuple{Tuple{Float64}}, (5.0, ), true],
        Any[Tuple{Tuple{Float64, Int}}, (5.0, 3), true],
        Any[Tuple{Float64, Tuple{Int}}, (5.0, 3), true],
        Any[Tuple{Float64, Tuple{Int, Float64}}, (5.0, 3, 4.0), true],
    ]
        ai = Taped.arginfo_from_argtypes(Tx, is_va)
        @test @inferred Taped.load_args!(ai, x) === nothing
    end

    @testset "TypedPhiNode" begin
        @testset "standard example of a phi node" begin
            node = TypedPhiNode(
                SlotRef{Float64}(),
                SlotRef{Float64}(),
                (1, 2),
                (ConstSlot(5.0), SlotRef(4.0)),
            )
            Taped.store_tmp_value!(node, 1)
            @test node.tmp_slot[] == 5.0
            Taped.transfer_tmp_value!(node)
            @test node.return_slot[] == 5.0
            Taped.store_tmp_value!(node, 2)
            @test node.tmp_slot[] == 4.0
            @test node.return_slot[] == 5.0
            Taped.transfer_tmp_value!(node)
            @test node.return_slot[] == 4.0
        end
        @testset "phi node with nothing in it" begin
            node = TypedPhiNode(SlotRef{Union{}}(), SlotRef{Union{}}(), (), ())
            Taped.store_tmp_value!(node, 1)
            Taped.transfer_tmp_value!(node)
        end
        @testset "phi node with undefined value" begin
            node = TypedPhiNode(
                SlotRef{Float64}(), SlotRef{Float64}(), (1, ), (SlotRef{Float64}(),)
            )
            Taped.store_tmp_value!(node, 1)
            Taped.transfer_tmp_value!(node)
        end
    end

    @testset "Unit tests for nodes and associated instructions" begin

        global __x_for_gref = 5.0
        global __y_for_gref::Float64 = 4.0

        @testset "ReturnNode" begin
            @testset "build_instruction: ReturnNode, $(Core.Typeof(args))" for args in Any[
                (SlotRef(5.0), SlotRef{Float64}()),
                (SlotRef(4), SlotRef{Any}()),
                (ConstSlot(5), SlotRef{Int}()),
                (ConstSlot(5.0), SlotRef{Real}()),
                (ConstSlot(:hi), SlotRef{Symbol}()),
                (ConstSlot(:hi), SlotRef{Any}()),
                (TypedGlobalRef(GlobalRef(Main, :__x_for_gref)), SlotRef{Any}()),
                (ConstSlot(sin), SlotRef{typeof(sin)}()),
            ]
                val, ret_slot = args
                oc = build_inst(ReturnNode, ret_slot, val)
                @test oc isa Taped.Inst
                output = oc(0)
                @test output == -1
                @test ret_slot[] == val[]
            end
        end

        @testset "GotoNode $label" for label in Any[1, 2, 3, 4, 5]
            oc = build_inst(GotoNode, label)
            @test oc isa Taped.Inst
            @test oc(3) == label
        end

        global __global_bool = false
        @testset "GotoIfNot $cond" for cond in Any[
            SlotRef(true), SlotRef(false),
            ConstSlot(true), ConstSlot(false),
            SlotRef{Any}(true), SlotRef{Real}(false),
            ConstSlot{Any}(true), ConstSlot{Any}(false),
            TypedGlobalRef(GlobalRef(Main, :__global_bool)),
        ]
            oc = build_inst(GotoIfNot, cond, 1, 2)
            @test oc isa Taped.Inst
            @test oc(5) == (cond[] ? 1 : 2)
        end

        global __global_bool = true
        @testset "PiNode" for (input, out, prev_blk, next_blk) in Any[
            (SlotRef{Any}(5.0), SlotRef{Float64}(), 2, 3),
            (ConstSlot{Float64}(5.0), SlotRef{Float64}(), 2, 2),
            (TypedGlobalRef(GlobalRef(Main, :__global_bool)), ConstSlot(true), 2, 2)
        ]
            oc = build_inst(PiNode, input, out, next_blk)
            @test oc isa Taped.Inst
            @test oc(prev_blk) == next_blk
            @test out[] == input[]
        end

        global __x_for_gref = 5.0
        @testset "GlobalRef" for (out, x, next_blk) in Any[
            (SlotRef{Float64}(), TypedGlobalRef(Main, :__x_for_gref), 5),
            (SlotRef{typeof(sin)}(), ConstSlot(sin), 4),
        ]
            oc = build_inst(GlobalRef, x, out, next_blk)
            @test oc isa Taped.Inst
            @test oc(4) == next_blk
            @test out[] == x[]
        end

        @testset "QuoteNode and literals" for (x, out, next_blk) in Any[
            (ConstSlot(5), SlotRef{Int}(), 5),
        ]
            oc = build_inst(nothing, x, out, next_blk)
            @test oc isa Taped.Inst
            @test oc(1) == next_blk
            @test out[] == x[]
        end

        @testset "Val{:boundscheck}" begin
            val_ref = SlotRef{Bool}()
            oc = build_inst(Val(:boundscheck), val_ref, 3)
            @test oc isa Taped.Inst
            @test oc(5) == 3
            @test val_ref[] == true
        end

        global __int_output = 5
        @testset "Val{:call}" for (arg_slots, evaluator, val_slot, next_blk) in Any[
            ((ConstSlot(sin), SlotRef(5.0)), Taped._eval, SlotRef{Float64}(), 3),
            ((ConstSlot(*), SlotRef(4.0), ConstSlot(4.0)), Taped._eval, SlotRef{Any}(), 3),
            (
                (ConstSlot(+), ConstSlot(4), ConstSlot(5)),
                Taped._eval,
                TypedGlobalRef(Main, :__int_output),
                2,
            ),
            (
                (ConstSlot(getfield), SlotRef((5.0, 5)), ConstSlot(1)),
                Taped.get_evaluator(
                    Taped.MinimalCtx(),
                    Tuple{typeof(getfield), Tuple{Float64, Int}, Int},
                    Expr(:call, ConstSlot(:getfield), SlotRef((5.0, 5)), ConstSlot(1)).args,
                    nothing,
                ),
                SlotRef{Float64}(),
                3,
            ),
        ]
            oc = build_inst(Val(:call), arg_slots, evaluator, val_slot, next_blk)
            @test oc isa Taped.Inst
            @test oc(0) == next_blk
            f, args... = map(getindex, arg_slots)
            @test val_slot[] == f(args...)
        end

        @testset "Val{:skipped_expression}" begin
            oc = build_inst(Val(:skipped_expression), 3)
            @test oc isa Taped.Inst
            @test oc(5) == 3
        end

        @testset "Val{:throw_undef_if_not}" begin
            @testset "defined" begin
                slot_to_check = SlotRef(5.0)
                oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
                @test oc isa Taped.Inst
                @test oc(0) == 2
            end
            @testset "undefined (non-isbits)" begin
                slot_to_check = SlotRef{Any}()
                oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
                @test oc isa Taped.Inst
                @test_throws ErrorException oc(3)
            end
            @testset "undefined (isbits)" begin
                slot_to_check = SlotRef{Float64}()
                oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
                @test oc isa Taped.Inst

                # a placeholder for failing to throw an ErrorException when evaluated
                @test_broken oc(5) == 1 
            end
        end
    end

    global __x_for_gref = 5.0
    @testset "_preprocess_expr_arg" begin
        sptypes = (Float64, )
        @test has_equal_data(_lift_expr_arg(Expr(:boundscheck), sptypes), ConstSlot(true))
        @test has_equal_data(
            _lift_expr_arg(Expr(:static_parameter, 1), sptypes), ConstSlot(Float64)
        )
        @test has_equal_data(_lift_expr_arg(5, sptypes), ConstSlot(5))
        @test has_equal_data(_lift_expr_arg(QuoteNode(:hello), sptypes), ConstSlot(:hello))
        @test has_equal_data(_lift_expr_arg(GlobalRef(Main, :sin), sptypes), ConstSlot(sin))
        @test has_equal_data(
            _lift_expr_arg(GlobalRef(Main, :__x_for_gref), sptypes),
            TypedGlobalRef(Main, :__x_for_gref),
        )
        @test has_equal_data(
            _lift_expr_arg(GlobalRef(Core.Intrinsics, :srem_int), sptypes),
            ConstSlot(Taped.IntrinsicsWrappers.srem_int),
        )
    end

    interp = Taped.TInterp()

    # nothings inserted for consistency with generate_test_functions.
    @testset "$f, $(map(Core.Typeof, x))" for (a, b, f, x...) in vcat(
        Any[
            (nothing, nothing, Taped.const_tester),
            (nothing, nothing, identity, 5.0),
            (nothing, nothing, Taped.foo, 5.0),
            (nothing, nothing, Taped.bar, 5.0, 4.0),
            (nothing, nothing, Taped.type_unstable_argument_eval, sin, 5.0),
            (nothing, nothing, Taped.pi_node_tester, Ref{Any}(5.0)),
            (nothing, nothing, Taped.pi_node_tester, Ref{Any}(5)),
            (nothing, nothing, Taped.intrinsic_tester, 5.0),
            (nothing, nothing, Taped.goto_tester, 5.0),
            (nothing, nothing, Taped.new_tester, 5.0, :hello),
            (nothing, nothing, Taped.new_tester_2, 4.0),
            (nothing, nothing, Taped.new_tester_3, Ref{Any}(Tuple{Float64})),
            (nothing, nothing, Taped.globalref_tester),
            (nothing, nothing, Taped.globalref_tester_2, true),
            (nothing, nothing, Taped.globalref_tester_2, false),
            (nothing, nothing, Taped.type_unstable_tester, Ref{Any}(5.0)),
            (nothing, nothing, Taped.type_unstable_tester_2, Ref{Real}(5.0)),
            (nothing, nothing, Taped.type_unstable_function_eval, Ref{Any}(sin), 5.0),
            (nothing, nothing, Taped.phi_const_bool_tester, 5.0),
            (nothing, nothing, Taped.phi_const_bool_tester, -5.0),
            (nothing, nothing, Taped.phi_node_with_undefined_value, true, 4.0),
            (nothing, nothing, Taped.phi_node_with_undefined_value, false, 4.0),
            (nothing, nothing, Taped.avoid_throwing_path_tester, 5.0),
            (nothing, nothing, Taped.simple_foreigncall_tester, randn(5)),
            (nothing, nothing, Taped.simple_foreigncall_tester_2, randn(6), (2, 3)),
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
            (nothing, nothing, Taped.inferred_const_tester, Ref{Any}(nothing)),
            (
                nothing,
                nothing,
                LinearAlgebra._modify!,
                LinearAlgebra.MulAddMul(5.0, 4.0),
                5.0,
                randn(5, 4),
                (5, 4),
            ), # for Bool comma,
            (nothing, nothing, Taped.getfield_tester, (5.0, 5)),
            (nothing, nothing, Taped.getfield_tester_2, (5.0, 5)),
            (
                nothing, nothing,
                mul!, transpose(randn(3, 5)), randn(5, 5), randn(5, 3), 4.0, 3.0,
            ), # static_parameter,
            (nothing, nothing, Xoshiro, 123456),
            (nothing, nothing, *, randn(250, 500), randn(500, 250)),
        ],
        TestResources.generate_test_functions(),
    )
        @info "$f, $(Core.Typeof(x))"
        sig = Tuple{Core.Typeof(f), map(Core.Typeof, x)...}
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp)

        # Verify correctness.
        @assert f(x...) == f(x...) # primal runs
        # @test TestUtils.has_equal_data(in_f(f, x...), f(x...))
        # @test TestUtils.has_equal_data(in_f(f, x...), f(x...)) # run twice to check for non-determinism.
        x_cpy_1 = deepcopy(x)
        x_cpy_2 = deepcopy(x)
        @test has_equal_data(in_f(f, x_cpy_1...), f(x_cpy_2...))
        @test has_equal_data(x_cpy_1, x_cpy_2)
        # TestUtils.test_rrule!!(
        #     Xoshiro(123456), in_f, x...;
        #     perf_flag=:none, interface_only=false, is_primitive=false,
        # )

        # # Taped.trace(f, deepcopy(x)...; ctx=Taped.RMC())
        # # rng = Xoshiro(123456)
        # # test_taped_rrule!!(rng, f, deepcopy(x)...; interface_only=false, perf_flag=:none)

        # # Only bother to check performance if the original programme does not allocate.
        # original = @benchmark $(Ref(f))[]($(Ref(x))[]...)
        # r = @benchmark $(Ref(in_f))[]($(Ref(f))[], $(Ref(x))[]...)

        # # __rrule!! = Taped.build_rrule!!(in_f)
        # # codual_x = map(zero_codual, x)
        # # rrule_timing = @benchmark($__rrule!!(zero_codual($in_f), $codual_x...))
        # # out, pb!! = __rrule!!(zero_codual(in_f), codual_x...)
        # # df = zero_codual(in_f)
        # # overall_timing = @benchmark Taped.to_benchmark($__rrule!!, $df, $codual_x)
        # println("original")
        # display(original)
        # println()
        # println("taped")
        # display(r)
        # println()
        # # println("rrule")
        # # display(rrule_timing)
        # # println()
        # # println("overall")
        # # display(overall_timing)
        # # println()

        # # if allocs(original) == 0
        # #     @test allocs(r) == 0
        # # end
    end
end
