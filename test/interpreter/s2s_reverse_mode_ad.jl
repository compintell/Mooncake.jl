@testset "s2s_reverse_mode_ad" begin
    @testset "make_ad_stmts!" begin
        info = ADInfo(ID(), LineToADDataMap())
        @testset "Nothing" begin
            @test TestUtils.has_equal_data(
                make_ad_stmts!(nothing, ID(), info),
                ADStmtInfo(nothing, nothing, nothing),
            )
        end
        @testset "ReturnNode" begin
            @test TestUtils.has_equal_data(
                make_ad_stmts!(ReturnNode(), ID(), info),
                ADStmtInfo(ReturnNode(), nothing, nothing),
            )
            @test TestUtils.has_equal_data(
                make_ad_stmts!(ReturnNode(5.0), ID(), info),
                ADStmtInfo(IDGotoNode(info.terminator_block_id), nothing, nothing),
            )
        end
        @testset "IDGotoNode" begin
            stmt = IDGotoNode(ID())
            @test TestUtils.has_equal_data(
                make_ad_stmts!(stmt, ID(), info), ADStmtInfo(stmt, nothing, nothing),
            )
        end
        @testset "IDGotoIfNot" begin
            stmt = IDGotoIfNot(ID(), ID())
            @test TestUtils.has_equal_data(
                make_ad_stmts!(stmt, ID(), info), ADStmtInfo(stmt, nothing, nothing),
            )
        end
        @testset "IDPhiNode" begin
            stmt = IDPhiNode(ID[ID(), ID()], Any[ID(), 5.0])
            @test TestUtils.has_equal_data(
                make_ad_stmts!(stmt, ID(), info), ADStmtInfo(stmt, nothing, nothing),
            )
        end
        @testset "GlobalRef" begin
            @test TestUtils.has_equal_data(
                make_ad_stmts!(GlobalRef(Base, :sin), ID(), info),
                ADStmtInfo(
                    Expr(:call, Taped.zero_codual, GlobalRef(Base, :sin)), nothing, nothing,
                ),
            )
        end
        @testset "QuoteNode" begin
            @test TestUtils.has_equal_data(
                make_ad_stmts!(QuoteNode("hi"), ID(), info),
                ADStmtInfo(QuoteNode(CoDual("hi", NoTangent())), nothing, nothing),
            )
        end
        @testset "literal" begin
            @test TestUtils.has_equal_data(
                make_ad_stmts!(5, ID(), info),
                ADStmtInfo(QuoteNode(CoDual(5, NoTangent())), nothing, nothing),
            )
            @test TestUtils.has_equal_data(
                make_ad_stmts!(0.0, ID(), info),
                ADStmtInfo(QuoteNode(CoDual(0.0, 0.0)), nothing, nothing),
            )
        end
        @testset "PhiCNode" begin
            @test_throws ErrorException make_ad_stmts!(Core.PhiCNode(Any[]), ID(), info)
        end
        @testset "UpsilonNode" begin
            @test_throws ErrorException make_ad_stmts!(Core.UpsilonNode(5), ID(), info)
        end
        @testset "Expr" begin
            @testset "throw_undef_if_not" begin
                cond_id = ID()
                expected_fwds = Expr(:call, Taped.__throw_undef_if_not, :x, cond_id)
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(Expr(:throw_undef_if_not, :x, cond_id), ID(), info),
                    ADStmtInfo(expected_fwds, nothing, nothing),
                )
            end
            @testset "$stmt" for stmt in [
                Expr(:boundscheck),
            ]
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(stmt, ID(), info), ADStmtInfo(stmt, nothing, nothing)
                )
            end
        end
    end

end
