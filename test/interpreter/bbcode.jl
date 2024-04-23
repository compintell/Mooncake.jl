module BBCodeTestCases
    test_phi_node(x::Ref{Union{Float32, Float64}}) = sin(x[])
end

@testset "bbcode" begin
    @testset "ID" begin
        id1 = ID()
        id2 = ID()
        @test id1 == id1
        @test id1 != id2
    end
    @testset "BBlock" begin
        bb = BBlock(
            ID(),
            ID[ID(), ID()],
            CC.NewInstruction[
                CC.NewInstruction(IDPhiNode([ID(), ID()], Any[true, false]), Any),
                CC.NewInstruction(:(println("hello")), Any),
            ],
        )
        @test bb isa BBlock
        @test length(bb) == 2

        ids, phi_nodes = Tapir.phi_nodes(bb)
        @test only(ids) == bb.inst_ids[1]
        @test only(phi_nodes) == bb.insts[1]

        insert!(bb, 1, ID(), CC.NewInstruction(nothing, Nothing))
        @test length(bb) == 3
        @test bb.insts[1].stmt === nothing

        bb_copy = copy(bb)
        @test bb_copy.inst_ids !== bb.inst_ids

        @test Tapir.terminator(bb) === nothing
    end
    @testset "BBCode $f" for (f, P) in [
        (TestResources.test_while_loop, Tuple{Float64}),
        (sin, Tuple{Float64}),
        (BBCodeTestCases.test_phi_node, Tuple{Ref{Union{Float32, Float64}}}),
    ]
        ir = Base.code_ircode(f, P)[1][1]
        bb_code = BBCode(ir)
        @test bb_code isa BBCode
        @test length(bb_code.blocks) == length(ir.cfg.blocks)
        new_ir = Tapir.IRCode(bb_code)
        @test length(new_ir.stmts.inst) == length(ir.stmts.inst)
        @test all(map(==, ir.stmts.inst, new_ir.stmts.inst))
        @test all(map(==, ir.stmts.type, new_ir.stmts.type))
        @test all(map(==, ir.stmts.info, new_ir.stmts.info))
        @test all(map(==, ir.stmts.line, new_ir.stmts.line))
        @test all(map(==, ir.stmts.flag, new_ir.stmts.flag))
        @test length(Tapir.collect_stmts(bb_code)) == length(ir.stmts.inst)
        @test Tapir.id_to_line_map(bb_code) isa Dict{ID, Int}
    end
    @testset "_characterise_unique_predecessor_blocks" begin
        @testset "single block" begin
            blk_id = ID()
            blks = BBlock[BBlock(blk_id, [ID()], [new_inst(ReturnNode(5))])]
            upreds, pred_is_upred = _characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id] == true
            @test pred_is_upred[blk_id] == true
        end
        @testset "pair of blocks" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blks = BBlock[
                BBlock(blk_id_1, [ID()], [new_inst(IDGotoNode(blk_id_2))]),
                BBlock(blk_id_2, [ID()], [new_inst(ReturnNode(5))]),
            ]
            upreds, pred_is_upred = _characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id_1] == true
            @test upreds[blk_id_2] == true
            @test pred_is_upred[blk_id_1] == true
            @test pred_is_upred[blk_id_2] == true
        end
        @testset "Non-Unique Exit Node" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blk_id_3 = ID()
            blks = BBlock[
                BBlock(blk_id_1, [ID()], [new_inst(IDGotoIfNot(true, blk_id_3))]),
                BBlock(blk_id_2, [ID()], [new_inst(ReturnNode(5))]),
                BBlock(blk_id_3, [ID()], [new_inst(ReturnNode(5))]),
            ]
            upreds, pred_is_upred = _characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id_1] == true
            @test upreds[blk_id_2] == false
            @test upreds[blk_id_3] == false
            @test pred_is_upred[blk_id_1] == true
            @test pred_is_upred[blk_id_2] == true
            @test pred_is_upred[blk_id_3] == true
        end
        @testset "diamond structure of four blocks" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blk_id_3 = ID()
            blk_id_4 = ID()
            blks = BBlock[
                BBlock(blk_id_1, [ID()], [new_inst(IDGotoIfNot(true, blk_id_3))]),
                BBlock(blk_id_2, [ID()], [new_inst(IDGotoNode(blk_id_4))]),
                BBlock(blk_id_3, [ID()], [new_inst(IDGotoNode(blk_id_4))]),
                BBlock(blk_id_4, [ID()], [new_inst(ReturnNode(0))]),
            ]
            upreds, pred_is_upred = _characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id_1] == true
            @test upreds[blk_id_2] == false
            @test upreds[blk_id_3] == false
            @test upreds[blk_id_4] == true
            @test pred_is_upred[blk_id_1] == true
            @test pred_is_upred[blk_id_2] == true
            @test pred_is_upred[blk_id_3] == true
            @test pred_is_upred[blk_id_4] == false
        end
        @testset "simple loop back to first block" begin
            blk_id_1 = ID()
            blk_id_2 = ID()
            blks = BBlock[
                BBlock(blk_id_1, [ID()], [new_inst(IDGotoIfNot(true, blk_id_1))]),
                BBlock(blk_id_2, [ID()], [new_inst(ReturnNode(5))]),
            ]
            upreds, pred_is_upred = _characterise_unique_predecessor_blocks(blks)
            @test upreds[blk_id_1] == true
            @test upreds[blk_id_2] == true
            @test pred_is_upred[blk_id_1] == false
            @test pred_is_upred[blk_id_2] == true
        end
    end
end
