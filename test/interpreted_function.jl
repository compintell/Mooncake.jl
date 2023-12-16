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

    @testset "preprocess_ir" begin
        _literals = Any[5, 5.0, 5f0, :hi, nothing, Int]
        part_defined_vals = Vector{Any}(undef, 3)
        part_defined_vals[1] = SSAValue(1)
        @testset "$x" for (x, target) in vcat(
            Any[
                # ReturnNodes
                (ReturnNode(5), ReturnNode(5)),
                (ReturnNode(Int), ReturnNode(Int)),
                (ReturnNode(SSAValue(5)), ReturnNode(SSAValue(5))),
                (ReturnNode(Argument(1)), ReturnNode(Argument(1))),

                # # GotoNode
                # (GotoNode(5), GotoNode(5)),

                # # GotoIfNot
                # (GotoIfNot(true, 5), GotoIfNot(Literal(true), 5)),
                # (GotoIfNot(false, 5), GotoIfNot(Literal(false), 5)),
                # (GotoIfNot(SSAValue(1), 4), GotoIfNot(SSAValue(1), 4)),
                # (GotoIfNot(Argument(1), 4), GotoIfNot(Argument(1), 4)),

                # # PhiNode
                # (
                #     PhiNode(Int32.(eachindex(_literals)), _literals),
                #     PhiNode(
                #         Int32.(eachindex(_literals)), Any[Literal(l) for l in _literals]
                #     ),
                # ),
                # (
                #     PhiNode(Int32[1, 2], Any[SSAValue(1), Argument(2)]),
                #     PhiNode(Int32[1, 2], Any[SSAValue(1), Argument(2)]),
                # ),
                # (
                #     PhiNode(Int32[1, 2], Vector{Any}(undef, 2)),
                #     PhiNode(Int32[1, 2], Vector{Any}(undef, 2)),
                # ),
                # (
                #     PhiNode(Int32[1, 2, 3], part_defined_vals),
                #     PhiNode(Int32[1, 2, 3], part_defined_vals),
                # ),

                # # PiNode
                # (PiNode(5, Int), PiNode(Literal(5), Int)),
                # (PiNode(SSAValue(3), Float64), PiNode(SSAValue(3), Float64)),

                # # GlobalRef
                # (GlobalRef(Main, :sin), TypedGlobalRef(sin)),

                # # QuoteNode
                # (QuoteNode(CartesianIndex(1, 1)), Literal(CartesianIndex(1, 1))),

                # # Expr
                # (Expr(:boundscheck), Literal(true)),
                # (
                #     Expr(:call, GlobalRef(Main, :sin), SSAValue(5)),
                #     Expr(:call, TypedGlobalRef(sin), SSAValue(5)),
                # ),
                # (
                #     Expr(:call, GlobalRef(Main, :sin), 5.0),
                #     Expr(:call, TypedGlobalRef(sin), Literal(5.0)),
                # ),
            ],
        )
            d = (sptypes=Core.Compiler.VarState, spnames=nothing)
            @test TestUtils.has_equal_data(preprocess_ir(x, d), target)
        end
    end

    @testset "Unit tests for nodes and associated instructions" begin

        extract_value(x::Taped.AbstractSlot) = x[]
        extract_value(x::QuoteNode) = x.value
        extract_value(x::GlobalRef) = Taped._get_globalref(x)
        extract_value(x) = x

        @testset "ReturnNode" begin
            @testset "build_instruction: ReturnNode, $(Core.Typeof(args))" for args in Any[
                (SlotRef(5.0), SlotRef{Float64}()),
                (SlotRef(4), SlotRef{Any}()),
                (ConstSlot(5), SlotRef{Int}()),
                (ConstSlot(5.0), SlotRef{Real}()),
                (5, SlotRef{Int}()),
                (5, SlotRef{Any}()),
                (5.0, SlotRef{Float64}()),
                (5.0, SlotRef{Any}()),
                (QuoteNode(:hi), SlotRef{Symbol}()),
                (QuoteNode(:hi), SlotRef{Any}()),
                (GlobalRef(Main, :sin), SlotRef{typeof(sin)}()),
                (GlobalRef(Main, :sin), SlotRef{Any}()),
            ]
                val, ret_slot = args
                oc = build_inst(ReturnNode, ret_slot, val)
                @test oc isa Taped.IFInstruction
                output = oc(0)
                @test output == -1
                @test ret_slot[] == extract_value(val)
            end
        end

        @testset "GotoNode $label" for label in Any[1, 2, 3, 4, 5]
            oc = build_inst(GotoNode, label)
            @test oc isa Taped.IFInstruction
            @test oc(3) == label
        end

        @testset "GotoIfNot $cond" for cond in Any[
            SlotRef(true), SlotRef(false), ConstSlot(true), ConstSlot(false),
        ]
            oc = build_inst(GotoIfNot, cond, 1, 2)
            @test oc isa Taped.IFInstruction
            @test oc(5) == (cond[] ? 1 : 2)
        end

        # dummy_pn = PhiNode(Int32[1], Any[5])
        # @testset "PhiNode $inst" for (inst, val, prev_blk, next_blk) in Any[
        #     (
        #         PhiNodeInst(
        #             (1, 2),
        #             (false, true),
        #             SlotRef{Bool}(),
        #             0,
        #             dummy_pn,
        #             1,
        #         ),
        #         false, 1, 0,
        #     ),
        #     (
        #         PhiNodeInst(
        #             (1, 2),
        #             (false, true),
        #             SlotRef{Bool}(),
        #             0,
        #             dummy_pn,
        #             1,
        #         ),
        #         true, 2, 0,
        #     ),
        #     (
        #         PhiNodeInst(
        #             (1, 2),
        #             (SlotRef(false), SlotRef{Bool}()),
        #             SlotRef{Bool}(),
        #             0,
        #             dummy_pn,
        #             1,
        #         ),
        #         false, 1, 0,
        #     ),
        #     (
        #         PhiNodeInst(
        #             (1, 2),
        #             (SlotRef{Bool}(), SlotRef(true)),
        #             SlotRef{Bool}(),
        #             0,
        #             dummy_pn,
        #             1,
        #         ),
        #         true, 2, 0,
        #     ),
        #     (
        #         PhiNodeInst(
        #             (1, 2),
        #             (SlotRef{Bool}(), SlotRef(true)),
        #             SlotRef{Bool}(),
        #             0,
        #             dummy_pn,
        #             1,
        #         ),
        #         true, 2, 0,
        #     ),
        #     (
        #         PhiNodeInst(
        #             (1, 2),
        #             (SlotRef{Bool}(), SlotRef{Bool}()),
        #             SlotRef(false),
        #             0,
        #             dummy_pn,
        #             1,
        #         ),
        #         false, 4, 0,
        #     ),
        #     (
        #         PhiNodeInst(
        #             (1, 2),
        #             (SlotRef{Bool}(), SlotRef{Bool}()),
        #             SlotRef(true),
        #             0,
        #             dummy_pn,
        #             1,
        #         ),
        #         true, 4, 0,
        #     ),
        # ]
        #     output = inst(prev_blk)
        #     @test inst.val_slot[] == val
        #     @test output == next_blk
        # end

        # TODO -- PiNodeInst tests

        # dummy_args = (0, :(sin(5.0)), 5)
        # @testset "CallInst $inst" for (inst, val) in Any[
        #     (CallInst((sin, 4.0), Taped._eval, SlotRef{Float64}(), dummy_args...), sin(4.0)),
        #     (CallInst((sin, 4.0), Taped._eval, SlotRef{Any}(), dummy_args...), sin(4.0)),
        #     (CallInst((sin, 5.0), Taped._eval, SlotRef{Float64}(), dummy_args...), sin(5.0)),
        #     (CallInst((sin, 5.0), Taped._eval, SlotRef{Real}(), dummy_args...), sin(5.0)),
        #     (CallInst((GlobalRef(Taped, :sin), SlotRef(5.0)), Taped._eval, SlotRef{Real}(), dummy_args...), sin(5.0)),
        # ]
        #     inst(0)
        #     @test inst.val_ref[] == val
        # end
    end

    # # nothings inserted for consistency with generate_test_functions.
    # @testset "$f, $(map(Core.Typeof, x))" for (a, b, f, x...) in vcat(
    #     Any[
    #         (nothing, nothing, Taped.const_tester),
    #         (nothing, nothing, identity, 5.0),
    #         # (nothing, nothing, Taped.foo, 5.0),
    #         # (nothing, nothing, Taped.bar, 5.0, 4.0),
    #         # (nothing, nothing, Taped.type_unstable_argument_eval, sin, 5.0),
    #         # (nothing, nothing, Taped.pi_node_tester, Ref{Any}(5.0)),
    #         # (nothing, nothing, Taped.pi_node_tester, Ref{Any}(5)),
    #         # (nothing, nothing, Taped.intrinsic_tester, 5.0),
    #         # (nothing, nothing, Taped.goto_tester, 5.0),
    #         # (nothing, nothing, Taped.new_tester, 5.0, :hello),
    #         # (nothing, nothing, Taped.new_tester_2, 4.0),
    #         # (nothing, nothing, Taped.new_tester_3, Ref{Any}(Tuple{Float64})),
    #         # (nothing, nothing, Taped.globalref_tester),
    #         # (nothing, nothing, Taped.globalref_tester_2, true),
    #         # (nothing, nothing, Taped.globalref_tester_2, false),
    #         # (nothing, nothing, Taped.type_unstable_tester, Ref{Any}(5.0)),
    #         # (nothing, nothing, Taped.type_unstable_tester_2, Ref{Real}(5.0)),
    #         # (nothing, nothing, Taped.type_unstable_function_eval, Ref{Any}(sin), 5.0),
    #         # (nothing, nothing, Taped.phi_const_bool_tester, 5.0),
    #         # (nothing, nothing, Taped.phi_const_bool_tester, -5.0),
    #         # (nothing, nothing, Taped.phi_node_with_undefined_value, true, 4.0),
    #         # (nothing, nothing, Taped.phi_node_with_undefined_value, false, 4.0),
    #         # (nothing, nothing, Taped.avoid_throwing_path_tester, 5.0),
    #         # (nothing, nothing, Taped.simple_foreigncall_tester, randn(5)),
    #         # (nothing, nothing, Taped.simple_foreigncall_tester_2, randn(6), (2, 3)),
    #         # (nothing, nothing, Taped.foreigncall_tester, randn(5)),
    #         # (nothing, nothing, Taped.no_primitive_inlining_tester, 5.0),
    #         # (nothing, nothing, Taped.varargs_tester, 5.0),
    #         # (nothing, nothing, Taped.varargs_tester, 5.0, 4),
    #         # (nothing, nothing, Taped.varargs_tester, 5.0, 4, 3.0),
    #         # (nothing, nothing, Taped.varargs_tester_2, 5.0),
    #         # (nothing, nothing, Taped.varargs_tester_2, 5.0, 4),
    #         # (nothing, nothing, Taped.varargs_tester_2, 5.0, 4, 3.0),
    #         # (nothing, nothing, Taped.varargs_tester_3, 5.0),
    #         # (nothing, nothing, Taped.varargs_tester_3, 5.0, 4),
    #         # (nothing, nothing, Taped.varargs_tester_3, 5.0, 4, 3.0),
    #         # (nothing, nothing, Taped.varargs_tester_4, 5.0),
    #         # (nothing, nothing, Taped.varargs_tester_4, 5.0, 4),
    #         # (nothing, nothing, Taped.varargs_tester_4, 5.0, 4, 3.0),
    #         # (nothing, nothing, Taped.splatting_tester, 5.0),
    #         # (nothing, nothing, Taped.splatting_tester, (5.0, 4.0)),
    #         # (nothing, nothing, Taped.splatting_tester, (5.0, 4.0, 3.0)),
    #         # # (nothing, nothing, Taped.unstable_splatting_tester, Ref{Any}(5.0)),
    #         # # (nothing, nothing, Taped.unstable_splatting_tester, Ref{Any}((5.0, 4.0))),
    #         # # (nothing, nothing, Taped.unstable_splatting_tester, Ref{Any}((5.0, 4.0, 3.0))),
    #         # (nothing, nothing, Taped.inferred_const_tester, Ref{Any}(nothing)),
    #         # (
    #         #     nothing,
    #         #     nothing,
    #         #     LinearAlgebra._modify!,
    #         #     LinearAlgebra.MulAddMul(5.0, 4.0),
    #         #     5.0,
    #         #     randn(5, 4),
    #         #     (5, 4),
    #         # ), # for Bool comma,
    #         # (
    #         #     nothing, nothing,
    #         #     mul!, transpose(randn(3, 5)), randn(5, 5), randn(5, 3), 4.0, 3.0,
    #         # ), # static_parameter,
    #         # (nothing, nothing, Xoshiro, 123456),
    #     ],
    #     # TestResources.generate_test_functions(),
    # )
    #     @info "$f, $(Core.Typeof(x))"
    #     sig = Tuple{Core.Typeof(f), map(Core.Typeof, x)...}
    #     in_f = Taped.InterpretedFunction(DefaultCtx(), sig; interp)

    #     # Verify correctness.
    #     @assert f(x...) == f(x...) # primal runs
    #     @test TestUtils.has_equal_data(in_f(f, x...), f(x...))
    #     @test TestUtils.has_equal_data(in_f(f, x...), f(x...)) # run twice to check for non-determinism.
    #     # TestUtils.test_rrule!!(
    #     #     Xoshiro(123456), in_f, x...;
    #     #     perf_flag=:none, interface_only=false, is_primitive=false,
    #     # )

    #     # Taped.trace(f, deepcopy(x)...; ctx=Taped.RMC())
    #     # rng = Xoshiro(123456)
    #     # test_taped_rrule!!(rng, f, deepcopy(x)...; interface_only=false, perf_flag=:none)

    #     # # Only bother to check performance if the original programme does not allocate.
    #     # original = @benchmark $(Ref(f))[]($(Ref(x))[]...)
    #     # r = @benchmark $(Ref(in_f))[]($(Ref(f))[], $(Ref(x))[]...)

    #     # # __rrule!! = Taped.build_rrule!!(in_f)
    #     # # codual_x = map(zero_codual, x)
    #     # # rrule_timing = @benchmark($__rrule!!(zero_codual($in_f), $codual_x...))
    #     # # out, pb!! = __rrule!!(zero_codual(in_f), codual_x...)
    #     # # df = zero_codual(in_f)
    #     # # overall_timing = @benchmark Taped.to_benchmark($__rrule!!, $df, $codual_x)
    #     # println("original")
    #     # display(original)
    #     # println()
    #     # println("taped")
    #     # display(r)
    #     # println()
    #     # # println("rrule")
    #     # # display(rrule_timing)
    #     # # println()
    #     # # println("overall")
    #     # # display(overall_timing)
    #     # # println()

    #     # # if allocs(original) == 0
    #     # #     @test allocs(r) == 0
    #     # # end
    # end
end
