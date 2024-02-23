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
            [(ID(), IDPhiNode([ID(), ID()], Any[true, false]))],
            [(ID(), :(println("hello")))],
            nothing,
        )
        @test bb isa BBlock
        @test !Taped.has_terminator(bb)
        @test length(bb) == 2

        @testset "concatenate_ids" begin
            @test Taped.concatenate_ids(bb) isa Vector{ID}
            @test length(Taped.concatenate_ids(bb)) == length(bb)
        end
        @testset "concatenate_stmts" begin
            @test Taped.concatenate_stmts(bb) isa Vector{Any}
            @test length(Taped.concatenate_stmts(bb)) == length(bb)
        end
        @test Taped.first_id(bb) == bb.phi_nodes[1][1]
    end
    @testset "BBCode $f" for f in [TestResources.test_while_loop, sin]
        ir = Base.code_ircode(f, Tuple{Float64})[1][1]
        bb_code = BBCode(ir)
        @test bb_code isa BBCode
        @test length(bb_code.blocks) == length(ir.cfg.blocks)
        new_ir = Taped.IRCode(bb_code)
        @test length(new_ir.stmts.inst) == length(ir.stmts.inst)
        @test all(map(==, ir.stmts.inst, new_ir.stmts.inst))
        @test length(Taped.collect_stmts(bb_code)) == length(ir.stmts.inst)
        @test Taped.id_to_line_map(bb_code) isa Dict{ID, Int}
    end
    @testset "make_unique_return_block" begin
        ir = Base.code_ircode(x::Int -> x > 0 ? 1 : 0, Tuple{Int})[1][1]
        bb_ir = BBCode(ir)
        unique_ret_bb_ir = Taped.make_unique_return_block(bb_ir)
        new_ir = Taped.IRCode(unique_ret_bb_ir)
        oc = Core.OpaqueClosure(ir; do_compile=true)
        new_oc = Core.OpaqueClosure(ir; do_compile=true)
        @test oc(1) == new_oc(1)
        @test oc(0) == new_oc(0)
    end
end
