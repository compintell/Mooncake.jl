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
            @test val_slot[] == ret_slot[]
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
        ret = SlotRef{CoDual{Float64, Float64}}()
        next_blk = 5
        fwds_inst, bwds_inst = build_coinsts(PiNode, val, ret, next_blk)

        # Test forwards instruction.
        @test fwds_inst isa Taped.FwdsInst
        @test fwds_inst(1) == next_blk
        @test primal(ret[]) == primal(val[])
        @test tangent(ret[]) == tangent(val[])

        # Test backwards instruction.
        @test bwds_inst isa Taped.BwdsInst
        ret[] = CoDual(5.0, 1.6)
        @test bwds_inst(3) == 3
        @test primal(val[]) == primal(ret[])
        @test tangent(val[]) == tangent(ret[])
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

    # @testset "Expr(:boundscheck)" begin
    #     val_ref = SlotRef{Bool}()
    #     oc = build_inst(Val(:boundscheck), val_ref, 3)
    #     @test oc isa Taped.Inst
    #     @test oc(5) == 3
    #     @test val_ref[] == true
    # end

    # global __int_output = 5
    # @testset "Expr(:call)" for (arg_slots, evaluator, val_slot, next_blk) in Any[
    #     ((ConstSlot(sin), SlotRef(5.0)), Taped._eval, SlotRef{Float64}(), 3),
    #     ((ConstSlot(*), SlotRef(4.0), ConstSlot(4.0)), Taped._eval, SlotRef{Any}(), 3),
    #     (
    #         (ConstSlot(+), ConstSlot(4), ConstSlot(5)),
    #         Taped._eval,
    #         TypedGlobalRef(Main, :__int_output),
    #         2,
    #     ),
    #     (
    #         (ConstSlot(getfield), SlotRef((5.0, 5)), ConstSlot(1)),
    #         Taped.get_evaluator(
    #             Taped.MinimalCtx(),
    #             Tuple{typeof(getfield), Tuple{Float64, Int}, Int},
    #             Expr(:call, ConstSlot(:getfield), SlotRef((5.0, 5)), ConstSlot(1)).args,
    #             nothing,
    #         ),
    #         SlotRef{Float64}(),
    #         3,
    #     ),
    # ]
    #     oc = build_inst(Val(:call), arg_slots, evaluator, val_slot, next_blk)
    #     @test oc isa Taped.Inst
    #     @test oc(0) == next_blk
    #     f, args... = map(getindex, arg_slots)
    #     @test val_slot[] == f(args...)
    # end

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
end
