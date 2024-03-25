array_ref_type(::Type{T}) where {T} = Base.RefArray{T, Vector{T}, Nothing}

@testset "reverse_mode_ad" begin

    # Testing specific nodes.
    @testset "ReturnNode" begin
        @testset "SlotRefs" begin
            ret = SlotRef{CoDual{Float64, Float64}}()
            ret_tangent = SlotRef{Float64}()
            val = SlotRef((CoDual(5.0, 1.0), top_ref(Stack(1.0))))
            fwds_inst, bwds_inst = build_coinsts(ReturnNode, ret, ret_tangent, val)

            # Test forwards instruction.
            @test fwds_inst isa Phi.FwdsInst
            @test fwds_inst(5) == -1
            @test ret[] == get_codual(val)
            @test (@allocations fwds_inst(5)) == 0
            
            # Test backwards instruction.
            @test bwds_inst isa Phi.BwdsInst
            ret_tangent[] = 2.0
            @test bwds_inst(5) isa Int
            @test get_tangent_stack(val)[] == 3.0
            @test (@allocations bwds_inst(5)) == 0
        end
        @testset "val slot is const" begin
            ret = SlotRef{CoDual{Float64, Float64}}()
            ret_tangent = SlotRef{Float64}()
            val = ConstSlot((CoDual(5.0, 0.0), top_ref(Stack(0.0))))
            fwds_inst, bwds_inst = build_coinsts(ReturnNode, ret, ret_tangent, val)

            # Test forwards instruction.
            @test fwds_inst isa Phi.FwdsInst
            @test fwds_inst(5) == -1
            @test ret[] == get_codual(val)
            @test (@allocations fwds_inst(5)) == 0

            # Test backwards instruction.
            @test bwds_inst isa Phi.BwdsInst
            @test bwds_inst(5) isa Int
            @test (@allocations bwds_inst(5)) == 0
        end
    end
    @testset "GotoNode" begin
        dest = 5
        fwds_inst, bwds_inst = build_coinsts(GotoNode, dest)

        # Test forwards instructions.
        @test fwds_inst isa Phi.FwdsInst
        @test fwds_inst(1) == dest
        @test (@allocations fwds_inst(1)) == 0

        # Test reverse instructions.
        @test bwds_inst isa Phi.BwdsInst
        @test bwds_inst(1) == 1
        @test (@allocations bwds_inst(1)) == 0
    end
    @testset "GotoIfNot" begin
        @testset "SlotRef cond" begin
            dest = 5
            next_blk = 3
            cond = SlotRef((zero_codual(true), NoTangentRef()))
            fwds_inst, bwds_inst = build_coinsts(GotoIfNot, dest, next_blk, cond)

            # Test forwards instructions.
            @test fwds_inst isa Phi.FwdsInst
            @test fwds_inst(1) == next_blk
            @test (@allocations fwds_inst(1)) == 0
            cond[] = (zero_codual(false), get_tangent_stack(cond))
            @test fwds_inst(1) == dest
            @test (@allocations fwds_inst(1)) == 0

            # Test backwards instructions.
            @test bwds_inst isa Phi.BwdsInst
            @test bwds_inst(4) == 4
            @test (@allocations bwds_inst(1)) == 0
        end
        @testset "ConstSlot" begin
            dest = 5
            next_blk = 3
            cond = ConstSlot((zero_codual(true), NoTangentRef()))
            fwds_inst, bwds_inst = build_coinsts(GotoIfNot, dest, next_blk, cond)

            # Test forwards instructions.
            @test fwds_inst isa Phi.FwdsInst
            @test fwds_inst(1) == next_blk
            @test (@allocations fwds_inst(1)) == 0

            # Test backwards instructions.
            @test bwds_inst isa Phi.BwdsInst
            @test bwds_inst(4) == 4
            @test (@allocations bwds_inst(1)) == 0
        end
    end
    @testset "TypedPhiNode" begin
        @testset "standard example of a phi node" begin
            nodes = (
                TypedPhiNode(
                    SlotRef{Tuple{CoDual{Float64, Float64}, Base.RefArray{Float64, Vector{Float64}, Nothing}}}(),
                    SlotRef{Tuple{CoDual{Float64, Float64}, Base.RefArray{Float64, Vector{Float64}, Nothing}}}(),
                    (1, 2),
                    (
                        ConstSlot((CoDual(5.0, 1.0), top_ref(Stack(1.0)))),
                        SlotRef((CoDual(4.0, 1.2), top_ref(Stack(1.2)))),
                    ),
                ),
                TypedPhiNode(
                    SlotRef{Tuple{CoDual{Union{}, NoTangent}, NoTangentRef}}(),
                    SlotRef{Tuple{CoDual{Union{}, NoTangent}, NoTangentRef}}(),
                    (),
                    (),
                ),
                TypedPhiNode(
                    SlotRef{Tuple{CoDual{Int, NoTangent}, NoTangentRef}}(),
                    SlotRef{Tuple{CoDual{Int, NoTangent}, NoTangentRef}}(),
                    (1, ),
                    (SlotRef{Tuple{CoDual{Int, NoTangent}, NoTangentRef}}(),), # undef element
                ),
            )
            next_blk = 0
            prev_blk = 1
            fwds_inst, bwds_inst = build_coinsts(Vector{PhiNode}, nodes, next_blk)

            # Test forwards instructions.
            @test fwds_inst isa Phi.FwdsInst
            @test fwds_inst(1) == next_blk
            @test (@allocations fwds_inst(1)) == 0
            @test nodes[1].tmp_slot[] == nodes[1].values[1][]
            @test nodes[1].ret_slot[] == nodes[1].tmp_slot[]
            @test !isassigned(nodes[2].tmp_slot)
            @test !isassigned(nodes[2].ret_slot)
            @test nodes[3].tmp_slot[] == nodes[3].values[1][]
            @test nodes[3].ret_slot[] == nodes[3].tmp_slot[]

            # Test backwards instructions.
            @test bwds_inst isa Phi.BwdsInst
            @test bwds_inst(4) == 4
            @test (@allocations bwds_inst(1)) == 0
        end
    end
    @testset "PiNode" begin
        val = SlotRef((CoDual{Any, Any}(5.0, 0.0), top_ref(Stack{Any}(0.0))))
        ret = SlotRef{Tuple{CoDual{Float64, Float64}, Base.RefArray{Float64, Vector{Float64}, Nothing}}}()
        next_blk = 5
        fwds_inst, bwds_inst = build_coinsts(PiNode, Float64, val, ret, next_blk)

        # Test forwards instruction.
        @test fwds_inst isa Phi.FwdsInst
        @test fwds_inst(1) == next_blk
        @test primal(get_codual(ret)) == primal(get_codual(val))
        @test tangent(get_codual(ret)) == tangent(get_codual(val))
        @test length(get_tangent_stack(ret)) == 1
        @test get_tangent_stack(ret)[] == tangent(get_codual(val))

        # Increment tangent associated to `val`. This is done in order to check that the
        # tangent to `val` is incremented on the reverse-pass, not replaced.
        Phi.increment_ref!(get_tangent_stack(val), 0.1)

        # Test backwards instruction.
        @test bwds_inst isa Phi.BwdsInst
        Phi.increment_ref!(get_tangent_stack(ret), 1.6)
        @test bwds_inst(3) == 3
        @test get_tangent_stack(val)[] == 1.6 + 0.1 # check increment has happened.
    end
    global __x_for_gref = 5.0
    @testset "GlobalRef" for (P, out, gref, next_blk) in Any[
        (
            Float64,
            SlotRef{Tuple{CoDual{Float64, Float64}, array_ref_type(Float64)}}(),
            TypedGlobalRef(Main, :__x_for_gref),
            5,
        ),
        (
            typeof(sin),
            SlotRef{Tuple{codual_type(typeof(sin)), NoTangentRef}}(),
            ConstSlot(sin),
            4,
        ),
    ]
        fwds_inst, bwds_inst = build_coinsts(GlobalRef, P, gref, out, next_blk)

        # Forwards pass.
        @test fwds_inst isa Phi.FwdsInst
        @test fwds_inst(4) == next_blk
        @test primal(get_codual(out)) == gref[]

        # Backwards pass.
        @test bwds_inst isa Phi.BwdsInst
        @test bwds_inst(10) == 10
    end
    @testset "QuoteNode and literals" for (x, out, next_blk) in Any[
        (
            ConstSlot(CoDual(5, NoTangent())),
            SlotRef{Tuple{CoDual{Int, NoTangent}, NoTangentRef}}(),
            5,
        ),
    ]
        fwds_inst, bwds_inst = build_coinsts(nothing, x, out, next_blk)

        @test fwds_inst isa Phi.FwdsInst
        @test fwds_inst(1) == next_blk
        @test get_codual(out) == x[]
        @test length(get_tangent_stack(out)) == 1
        @test get_tangent_stack(out)[] == tangent(get_codual(out))

        @test bwds_inst isa Phi.BwdsInst
        @test bwds_inst(10) == 10
    end

    @testset "Expr(:boundscheck)" begin
        val_ref = SlotRef{Tuple{codual_type(Bool), NoTangentRef}}()
        next_blk = 3
        fwds_inst, bwds_inst = build_coinsts(Val(:boundscheck), val_ref, next_blk)

        @test fwds_inst isa Phi.FwdsInst
        @test fwds_inst(0) == next_blk
        @test get_codual(val_ref) == zero_codual(true)
        @test length(get_tangent_stack(val_ref)) == 1
        @test bwds_inst isa Phi.BwdsInst
        @test bwds_inst(2) == 2
    end

    global __int_output = 5
    @testset "Expr(:call)" for (P, out, arg_slots, next_blk) in Any[
        (
            Float64,
            SlotRef{Tuple{codual_type(Float64), array_ref_type(Float64)}}(),
            (
                ConstSlot((zero_codual(sin), top_ref(Stack(zero_tangent(sin))))),
                SlotRef((zero_codual(5.0), top_ref(Stack(0.0)))),
            ),
            3,
        ),
        (
            Any,
            SlotRef{Tuple{CoDual, array_ref_type(Any)}}(),
            (
                ConstSlot((zero_codual(*), top_ref(Stack(zero_tangent(*))))),
                SlotRef((zero_codual(4.0), top_ref(Stack(0.0)))),
                ConstSlot((zero_codual(4.0), top_ref(Stack(0.0)))),
            ),
            3,
        ),
        (
            Int,
            SlotRef{Tuple{codual_type(Int), NoTangentRef}}(),
            (
                ConstSlot((zero_codual(+), top_ref(Stack(zero_tangent(+))))),
                ConstSlot((zero_codual(4), NoTangentRef())),
                ConstSlot((zero_codual(5), NoTangentRef())),
            ),
            2,
        ),
        (
            Float64,
            SlotRef{Tuple{codual_type(Float64), array_ref_type(Float64)}}(),    
            (
                ConstSlot((zero_codual(getfield), NoTangentRef())),
                SlotRef((zero_codual((5.0, 5)), top_ref(Stack(zero_tangent((5.0, 5)))))),
                ConstSlot((zero_codual(1), NoTangentRef())),
            ),
            3,
        ),
    ]
        sig = _typeof(map(primal ∘ get_codual, arg_slots))
        interp = Phi.PInterp()
        evaluator = Phi.get_evaluator(Phi.MinimalCtx(), sig, interp, true)
        __rrule!! = Phi.get_rrule!!_evaluator(evaluator)
        pb_stack = Phi.build_pb_stack(__rrule!!, evaluator, arg_slots)
        fwds_inst, bwds_inst = build_coinsts(
            Val(:call), P, out, arg_slots, evaluator, __rrule!!, pb_stack, next_blk
        )

        # Test forwards-pass.
        @test fwds_inst isa Phi.FwdsInst
        @test fwds_inst(0) == next_blk

        # Test reverse-pass.
        @test bwds_inst isa Phi.BwdsInst
        @test bwds_inst(5) == 5
    end

    @testset "Expr(:skipped_expression)" begin
        next_blk = 3
        fwds_inst, bwds_inst = build_coinsts(Val(:skipped_expression), next_blk)

        # Test forwards pass.
        @test fwds_inst isa Phi.FwdsInst
        @test fwds_inst(1) == next_blk

        # Test backwards pass.
        @test bwds_inst isa Phi.BwdsInst
    end

    # @testset "Expr(:throw_undef_if_not)" begin
    #     @testset "defined" begin
    #         slot_to_check = SlotRef(5.0)
    #         oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
    #         @test oc isa Phi.Inst
    #         @test oc(0) == 2
    #     end
    #     @testset "undefined (non-isbits)" begin
    #         slot_to_check = SlotRef{Any}()
    #         oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
    #         @test oc isa Phi.Inst
    #         @test_throws ErrorException oc(3)
    #     end
    #     @testset "undefined (isbits)" begin
    #         slot_to_check = SlotRef{Float64}()
    #         oc = build_inst(Val(:throw_undef_if_not), slot_to_check, 2)
    #         @test oc isa Phi.Inst

    #         # a placeholder for failing to throw an ErrorException when evaluated
    #         @test_broken oc(5) == 1 
    #     end
    # end

    interp = Phi.PInterp()

    # nothings inserted for consistency with generate_test_functions.
    @testset "$(_typeof((f, x...)))" for (interface_only, perf_flag, bnds, f, x...) in
        TestResources.generate_test_functions()

        sig = _typeof((f, x...))
        @info "$sig"
        in_f = Phi.InterpretedFunction(DefaultCtx(), sig, interp);

        # Verify correctness.
        @assert f(deepcopy(x)...) == f(deepcopy(x)...) # primal runs
        x_cpy_1 = deepcopy(x)
        x_cpy_2 = deepcopy(x)
        @test has_equal_data(in_f(f, x_cpy_1...), f(x_cpy_2...))
        @test has_equal_data(x_cpy_1, x_cpy_2)
        rule = Phi.build_rrule!!(in_f);
        TestUtils.test_rrule!!(
            Xoshiro(123456), in_f, f, x...;
            perf_flag, interface_only, is_primitive=false, rule
        )

        # # Estimate primal performance.
        # original = @benchmark $(Ref(f))[]($(Ref(deepcopy(x)))[]...);

        # # Estimate interpretered function performance.
        # r = @benchmark $(Ref(in_f))[]($(Ref(f))[], $(Ref(deepcopy(x)))[]...);

        # # Estimate overal forwards-pass and pullback performance.
        # __rrule!! = Phi.build_rrule!!(in_f);
        # df = zero_codual(in_f);
        # codual_x = map(zero_codual, (f, x...));
        # overall_timing = @benchmark TestUtils.to_benchmark($__rrule!!, $df, $codual_x...);

        # # Print the results.
        # println("original")
        # display(original)
        # println()
        # println("phi")
        # display(r)
        # println()
        # println("overall")
        # display(overall_timing)
        # println()

        # @profview run_many_times(10, TestUtils.to_benchmark, __rrule!!, df, codual_x)
    end
end
