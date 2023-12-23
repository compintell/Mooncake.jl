@testset "reverse_mode_ad" begin
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
end
