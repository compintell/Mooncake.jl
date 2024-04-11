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
end
