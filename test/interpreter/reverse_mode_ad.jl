@testset "reverse_mode_ad" begin

    # Testing specific nodes.
    @testset "ReturnNode" begin
        @testset "SlotRefs" begin
            ret_slot = SlotRef{CoDual{Float64, Float64}}()
            val_slot = SlotRef(CoDual(5.0, 1.0))
            fwds_inst, bwds_inst = build_coinsts(ReturnNode, ret_slot, val_slot)

            # Test forwards instruction.
            @test fwds_inst isa Taped.FwdsInst
            @test fwds_inst(5) == -1
            @test ret_slot[] == val_slot[]
            @test (@allocations fwds_inst(5)) == 0
            
            # Test backwards instruction.
            @test bwds_inst isa Taped.BwdsInst
            ret_slot[] = CoDual(5.0, 2.0)
            @test bwds_inst(5) isa Int
            @test val_slot[] == CoDual(5.0, 3.0)
            @test (@allocations bwds_inst(5)) == 0
        end
        @testset "val slot is const" begin
            ret_slot = SlotRef{CoDual{Float64, Float64}}()
            val_slot = ConstSlot(CoDual(5.0, 0.0))
            fwds_inst, bwds_inst = build_coinsts(ReturnNode, ret_slot, val_slot)

            # Test forwards instruction.
            @test fwds_inst isa Taped.FwdsInst
            @test fwds_inst(5) == -1
            @test ret_slot[] == val_slot[]
            @test (@allocations fwds_inst(5)) == 0

            # Test backwards instruction.
            @test bwds_inst isa Taped.BwdsInst
            @test bwds_inst(5) isa Int
            @test (@allocations bwds_inst(5)) == 0
        end
    end
    @testset "GotoNode" begin
        dest = 5
        fwds_inst, bwds_inst = build_coinsts(GotoNode, dest)

        # Test forwards instructions.
        @test fwds_inst isa Taped.FwdsInst
        @test fwds_inst(1) == dest
        @test (@allocations fwds_inst(1)) == 0

        # Test reverse instructions.
        @test bwds_inst isa Taped.BwdsInst
        @test bwds_inst(1) == 1
        @test (@allocations bwds_inst(1)) == 0
    end
    @testset "GotoIfNot" begin
        @testset "SlotRef cond" begin
            dest = 5
            next_blk = 3
            cond = SlotRef(zero_codual(true))
            fwds_inst, bwds_inst = build_coinsts(GotoIfNot, dest, next_blk, cond)

            # Test forwards instructions.
            @test fwds_inst isa Taped.FwdsInst
            @test fwds_inst(1) == next_blk
            @test (@allocations fwds_inst(1)) == 0
            cond[] = zero_codual(false)
            @test fwds_inst(1) == dest
            @test (@allocations fwds_inst(1)) == 0

            # Test backwards instructions.
            @test bwds_inst isa Taped.BwdsInst
            @test bwds_inst(4) == 4
            @test (@allocations bwds_inst(1)) == 0
        end
        @testset "ConstSlot" begin
            dest = 5
            next_blk = 3
            cond = ConstSlot(zero_codual(true))
            fwds_inst, bwds_inst = build_coinsts(GotoIfNot, dest, next_blk, cond)

            # Test forwards instructions.
            @test fwds_inst isa Taped.FwdsInst
            @test fwds_inst(1) == next_blk
            @test (@allocations fwds_inst(1)) == 0

            # Test backwards instructions.
            @test bwds_inst isa Taped.BwdsInst
            @test bwds_inst(4) == 4
            @test (@allocations bwds_inst(1)) == 0
        end
    end
    @testset "TypedPhiNode" begin
        @testset "standard example of a phi node" begin
            nodes = (
                TypedPhiNode(
                    SlotRef{CoDual{Float64, Float64}}(),
                    SlotRef{CoDual{Float64, Float64}}(),
                    (1, 2),
                    (ConstSlot(CoDual(5.0, 1.0)), SlotRef(CoDual(4.0, 1.2))),
                ),
                TypedPhiNode(
                    SlotRef{CoDual{Union{}, NoTangent}}(),
                    SlotRef{CoDual{Union{}, NoTangent}}(),
                    (),
                    (),
                ),
                TypedPhiNode(
                    SlotRef{CoDual{Int, NoTangent}}(),
                    SlotRef{CoDual{Int, NoTangent}}(),
                    (1, ),
                    (SlotRef{CoDual{Int, NoTangent}}(),),
                ),
            )
            next_blk = 0
            prev_blk = 1
            fwds_inst, bwds_inst = build_coinsts(
                Vector{PhiNode}, nodes, next_blk, Taped.make_stacks(nodes)...
            )

            # Test forwards instructions.
            @test fwds_inst isa Taped.FwdsInst
            @test fwds_inst(1) == next_blk
            @test (@allocations fwds_inst(1)) == 0
            @test nodes[1].tmp_slot[] == nodes[1].values[1][]
            @test nodes[1].ret_slot[] == nodes[1].tmp_slot[]
            @test !isassigned(nodes[2].tmp_slot)
            @test !isassigned(nodes[2].ret_slot)
            @test nodes[3].tmp_slot[] == nodes[3].values[1][]
            @test nodes[3].ret_slot[] == nodes[3].tmp_slot[]

            # Test backwards instructions.
            @test bwds_inst isa Taped.BwdsInst
            @test bwds_inst(4) == 4
            @test (@allocations bwds_inst(1)) == 0
        end
    end
    @testset "PiNode" begin
        val = SlotRef{CoDual{Any, Any}}(CoDual{Any, Any}(5.0, 0.0))
        ret = SlotRef{CoDual{Float64, Float64}}(CoDual(-1.0, -1.0))
        old_vals = Vector{CoDual{Float64, Float64}}(undef, 0)
        next_blk = 5
        fwds_inst, bwds_inst = build_coinsts(PiNode, val, ret, old_vals, next_blk)

        # Test forwards instruction.
        @test fwds_inst isa Taped.FwdsInst
        @test fwds_inst(1) == next_blk
        @test primal(ret[]) == primal(val[])
        @test tangent(ret[]) == tangent(val[])

        # Increment tangent associated to `val`. This is done in order to check that the
        # tangent to `val` is incremented on the reverse-pass, not replaced.
        val[] = CoDual{Any, Any}(primal(val[]), tangent(val[]) + 0.1)

        # Test backwards instruction.
        @test bwds_inst isa Taped.BwdsInst
        ret[] = CoDual(5.0, 1.6)
        @test bwds_inst(3) == 3
        @test primal(ret[]) == -1.0
        @test tangent(ret[]) == -1.0
        @test tangent(val[]) == 1.6 + 0.1 # check increment has happened.
    end
    global __x_for_gref = 5.0
    @testset "GlobalRef" for (out, x, next_blk) in Any[
        (SlotRef{CoDual{Float64, Float64}}(), TypedGlobalRef(Main, :__x_for_gref), 5),
        (SlotRef{CoDual{typeof(sin), tangent_type(typeof(sin))}}(), ConstSlot(sin), 4),
    ]
        fwds_inst, bwds_inst = build_coinsts(GlobalRef, x, out, next_blk)

        # Forwards pass.
        @test fwds_inst isa Taped.FwdsInst
        @test fwds_inst(4) == next_blk
        @test primal(out[]) == x[]

        # Backwards pass.
        @test bwds_inst isa Taped.BwdsInst
        @test bwds_inst(10) == 10
    end
    @testset "QuoteNode and literals" for (x, out, next_blk) in Any[
        (ConstSlot(CoDual(5, NoTangent())), SlotRef{CoDual{Int, NoTangent}}(), 5),
    ]
        fwds_inst, bwds_inst = build_coinsts(nothing, x, out, next_blk)

        @test fwds_inst isa Taped.FwdsInst
        @test fwds_inst(1) == next_blk
        @test out[] == x[]

        @test bwds_inst isa Taped.BwdsInst
        @test bwds_inst(10) == 10
    end

    @testset "Expr(:boundscheck)" begin
        val_ref = SlotRef{codual_type(Bool)}()
        next_blk = 3
        fwds_inst, bwds_inst = build_coinsts(Val(:boundscheck), val_ref, next_blk)

        @test fwds_inst isa Taped.FwdsInst
        @test fwds_inst(0) == next_blk
        @test val_ref[] == zero_codual(true)
        @test bwds_inst isa Taped.BwdsInst
        @test bwds_inst(2) == 2
    end

    global __int_output = 5
    @testset "Expr(:call)" for (out, arg_slots, next_blk) in Any[
        (
            SlotRef{codual_type(Float64)}(),
            (ConstSlot(zero_codual(sin)), SlotRef(zero_codual(5.0))),
            3,
        ),
        (
            SlotRef{CoDual}(),
            (
                ConstSlot(zero_codual(*)),
                SlotRef(zero_codual(4.0)),
                ConstSlot(zero_codual(4.0)),
            ),
            3,
        ),
        (
            SlotRef{codual_type(Int)}(),
            (
                ConstSlot(zero_codual(+)),
                ConstSlot(zero_codual(4)),
                ConstSlot(zero_codual(5)),
            ),
            2,
        ),
        (
            SlotRef{codual_type(Float64)}(),    
            (
                ConstSlot(zero_codual(getfield)),
                SlotRef(zero_codual((5.0, 5))),
                ConstSlot(zero_codual(1)),
            ),
            3,
        ),
    ]
        sig = Tuple{map(Core.Typeof ∘ primal ∘ getindex, arg_slots)...}
        interp = Taped.TInterp()
        evaluator = Taped.get_evaluator(Taped.MinimalCtx(), sig, nothing, interp)
        __rrule!! = Taped.get_rrule!!_evaluator(evaluator)
        old_vals = Vector{eltype(out)}(undef, 0)
        pb_stack = Taped.build_pb_stack(__rrule!!, evaluator, arg_slots)
        fwds_inst, bwds_inst = build_coinsts(
            Val(:call), out, arg_slots, evaluator, __rrule!!, old_vals, pb_stack, next_blk
        )

        # Test forwards-pass.
        @test fwds_inst isa Taped.FwdsInst
        @test fwds_inst(0) == next_blk

        # Test reverse-pass.
        @test bwds_inst isa Taped.BwdsInst
        @test bwds_inst(5) == 5
    end

    @testset "Expr(:skipped_expression)" begin
        next_blk = 3
        fwds_inst, bwds_inst = build_coinsts(Val(:skipped_expression), next_blk)

        # Test forwards pass.
        @test fwds_inst isa Taped.FwdsInst
        @test fwds_inst(1) == next_blk

        # Test backwards pass.
        @test bwds_inst isa Taped.BwdsInst
    end

    # @testset "Expr(:throw_undef_if_not)" begin
    #     @testset "defined" begin
    #         slot_to_check = SlotRef(5.0)
    #         oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
    #         @test oc isa Taped.Inst
    #         @test oc(0) == 2
    #     end
    #     @testset "undefined (non-isbits)" begin
    #         slot_to_check = SlotRef{Any}()
    #         oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
    #         @test oc isa Taped.Inst
    #         @test_throws ErrorException oc(3)
    #     end
    #     @testset "undefined (isbits)" begin
    #         slot_to_check = SlotRef{Float64}()
    #         oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
    #         @test oc isa Taped.Inst

    #         # a placeholder for failing to throw an ErrorException when evaluated
    #         @test_broken oc(5) == 1 
    #     end
    # end

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
            # (nothing, nothing, Taped.globalref_tester_2, true),
            # (nothing, nothing, Taped.globalref_tester_2, false),
            (nothing, nothing, Taped.type_unstable_tester, Ref{Any}(5.0)),
            (nothing, nothing, Taped.type_unstable_tester_2, Ref{Real}(5.0)),
            (nothing, nothing, Taped.type_unstable_tester_3, Ref{Any}(5.0)),
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
            # (nothing, nothing, Taped.unstable_splatting_tester, Ref{Any}(5.0)), # known failure case -- no rrule for _apply_iterate
            # (nothing, nothing, Taped.unstable_splatting_tester, Ref{Any}((5.0, 4.0))), # known failure case -- no rrule for _apply_iterate
            # (nothing, nothing, Taped.unstable_splatting_tester, Ref{Any}((5.0, 4.0, 3.0))), # known failure case -- no rrule for _apply_iterate
            (nothing, nothing, Taped.inferred_const_tester, Ref{Any}(nothing)),
            (nothing, nothing, Taped.datatype_slot_tester, 1),
            (nothing, nothing, Taped.datatype_slot_tester, 2),
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
        in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp);

        # Verify correctness.
        @assert f(deepcopy(x)...) == f(deepcopy(x)...) # primal runs
        x_cpy_1 = deepcopy(x)
        x_cpy_2 = deepcopy(x)
        @test has_equal_data(in_f(f, x_cpy_1...), f(x_cpy_2...))
        @test has_equal_data(x_cpy_1, x_cpy_2)
        TestUtils.test_rrule!!(
            Xoshiro(123456), in_f, f, x...;
            perf_flag=:none, interface_only=false, is_primitive=false,
        )

        # # Helper code for debugging.
        # rule = Taped.build_rrule!!(in_f)
        # args = map(zero_codual, (in_f, f, x...))
        # out, pb!! = rule(args...)
        # pb!!(tangent(out), map(tangent, args)...)

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
