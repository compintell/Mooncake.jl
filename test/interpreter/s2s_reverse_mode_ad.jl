module S2SGlobals
    non_const_global = 5.0
    const const_float = 5.0
    const const_int = 5
    const const_bool = true
end

@testset "s2s_reverse_mode_ad" begin
    @testset "SharedDataPairs" begin
        m = SharedDataPairs()
        id = Taped.add_data!(m, 5.0)
        @test length(m.pairs) == 1
        @test m.pairs[1][1] == id
        @test m.pairs[1][2] == 5.0
    end
    @testset "ADInfo" begin
        arg_types = Dict{Argument, Any}(Argument(1) => Float64, Argument(2) => Int)
        id_ssa_1 = ID()
        id_ssa_2 = ID()
        ssa_insts = Dict{ID, CC.NewInstruction}(
            id_ssa_1 => CC.NewInstruction(nothing, Float64),
            id_ssa_2 => CC.NewInstruction(nothing, Any),
        )
        info = ADInfo(Taped.TInterp(), arg_types, ssa_insts, Any[])

        # Verify that we can access the interpreter and terminator block ID.
        @test info.interp isa Taped.TInterp

        # Verify that we can get the type associated to Arguments, IDs, and others.
        global ___x = 5.0
        global ___y::Float64 = 5.0
        @test Taped.get_primal_type(info, Argument(1)) == Float64
        @test Taped.get_primal_type(info, Argument(2)) == Int
        @test Taped.get_primal_type(info, id_ssa_1) == Float64
        @test Taped.get_primal_type(info, GlobalRef(Base, :sin)) == typeof(sin)
        @test Taped.get_primal_type(info, GlobalRef(Main, :___x)) == Any
        @test Taped.get_primal_type(info, GlobalRef(Main, :___y)) == Float64
        @test Taped.get_primal_type(info, 5) == Int
        @test Taped.get_primal_type(info, QuoteNode(:hello)) == Symbol
    end
    @testset "make_ad_stmts!" begin

        # Set up ADInfo -- this state is required by `make_ad_stmts!`, and the
        # `LineToADDataMap` object can be mutated.
        id_line_1 = ID()
        id_line_2 = ID()
        info = ADInfo(
            Taped.TInterp(),
            Dict{Argument, Any}(Argument(1) => typeof(sin), Argument(2) => Float64),
            Dict{ID, CC.NewInstruction}(
                id_line_1 => CC.NewInstruction(Expr(:invoke, nothing, cos, Argument(2)), Float64),
                id_line_2 => CC.NewInstruction(nothing, Any),
            ),
            Any[Taped.NoTangentStack(), Stack{Float64}()],
        )

        @testset "Nothing" begin
            line = ID()
            @test TestUtils.has_equal_data(
                make_ad_stmts!(nothing, line, info),
                ad_stmt_info(line, nothing, nothing),
            )
        end
        @testset "ReturnNode" begin
            line = ID()
            @testset "unreachable" begin
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(ReturnNode(), line, info),
                    ad_stmt_info(line, ReturnNode(), nothing),
                )
            end
            @testset "Argument" begin
                val = Argument(4)
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(ReturnNode(Argument(2)), line, info),
                    ad_stmt_info(line, ReturnNode(Argument(3)), nothing),
                )
            end
            @testset "literal" begin
                stmt_info = make_ad_stmts!(ReturnNode(5.0), line, info)
                @test stmt_info isa ADStmtInfo
                @test stmt_info.fwds[1][2].stmt isa ReturnNode
            end
            @testset "GlobalRef" begin
                node = ReturnNode(GlobalRef(S2SGlobals, :const_float))
                stmt_info = make_ad_stmts!(node, line, info)
                @test stmt_info isa ADStmtInfo
                @test stmt_info.fwds[1][2].stmt isa ReturnNode
            end
        end
        @testset "IDGotoNode" begin
            line = ID()
            stmt = IDGotoNode(ID())
            @test TestUtils.has_equal_data(
                make_ad_stmts!(stmt, line, info), ad_stmt_info(line, stmt, nothing)
            )
        end
        @testset "IDGotoIfNot" begin
            line = ID()
            cond_id = ID()
            stmt = IDGotoIfNot(cond_id, ID())
            ad_stmts = make_ad_stmts!(stmt, line, info)
            @test ad_stmts isa ADStmtInfo
            @test ad_stmts.rvs[1][2].stmt === nothing
            fwds = ad_stmts.fwds
            @test fwds[1][1] == fwds[2][2].stmt.cond
            @test Meta.isexpr(fwds[1][2].stmt, :call)
            @test fwds[2][2].stmt isa IDGotoIfNot
            @test fwds[2][2].stmt.dest == stmt.dest
        end
        @testset "IDPhiNode" begin
            line = ID()
            stmt = IDPhiNode(ID[ID(), ID()], Any[ID(), 5.0])
            ad_stmts = make_ad_stmts!(stmt, line, info)
            @test ad_stmts isa ADStmtInfo
        end
        @testset "PiNode" begin
            @testset "unhandled case" begin
                @test_throws(
                    Taped.UnhandledLanguageFeatureException,
                    make_ad_stmts!(PiNode(5.0, Float64), ID(), info),
                )
            end
            @testset "sharpen type of ID" begin
                line = ID()
                val = id_line_2
                stmt_info = make_ad_stmts!(PiNode(val, Float64), line, info)
                @test stmt_info isa ADStmtInfo
            end
        end
        @testset "GlobalRef" begin
            @testset "non-const" begin
                global_ref = GlobalRef(S2SGlobals, :non_const_global)
                stmt_info = make_ad_stmts!(global_ref, ID(), info)
                @test stmt_info isa Taped.ADStmtInfo
                @test Meta.isexpr(last(stmt_info.fwds)[2].stmt, :call)
                @test last(stmt_info.fwds)[2].stmt.args[1] == Taped.__verify_const
            end
            @testset "differentiable const globals" begin
                stmt_info = make_ad_stmts!(GlobalRef(S2SGlobals, :const_float), ID(), info)
                @test stmt_info isa Taped.ADStmtInfo
                @test Meta.isexpr(only(stmt_info.fwds)[2].stmt, :call)
                @test only(stmt_info.fwds)[2].stmt.args[1] == identity
            end
        end
        @testset "PhiCNode" begin
            @test_throws(
                Taped.UnhandledLanguageFeatureException,
                make_ad_stmts!(Core.PhiCNode(Any[]), ID(), info),
            )
        end
        @testset "UpsilonNode" begin
            @test_throws(
                Taped.UnhandledLanguageFeatureException,
                make_ad_stmts!(Core.UpsilonNode(5), ID(), info),
            )
        end
        @testset "Expr" begin
            @testset "invoke" begin
                stmt = Expr(:invoke, nothing, cos, Argument(2))
                ad_stmts = make_ad_stmts!(stmt, id_line_1, info)
                fwds_stmt = ad_stmts.fwds[2][2].stmt
                @test Meta.isexpr(fwds_stmt, :call)
                @test fwds_stmt.args[1] == Taped.__fwds_pass!
                @test Meta.isexpr(ad_stmts.rvs[2][2].stmt, :call)
                @test ad_stmts.rvs[2][2].stmt.args[1] == Taped.__rvs_pass!
            end
            @testset "copyast" begin
                stmt = Expr(:copyast, QuoteNode(:(hi)))
                ad_stmts = make_ad_stmts!(stmt, ID(), info)
                @test ad_stmts isa Taped.ADStmtInfo
                @test Meta.isexpr(ad_stmts.fwds[1][2].stmt, :call)
                @test ad_stmts.fwds[1][2].stmt.args[1] == identity
            end
            @testset "throw_undef_if_not" begin
                cond_id = ID()
                line = ID()
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(Expr(:throw_undef_if_not, :x, cond_id), line, info),
                    ad_stmt_info(line, Expr(:throw_undef_if_not, :x, cond_id), nothing),
                )
            end
            @testset "$stmt" for stmt in [
                Expr(:gc_preserve_begin),
            ]
                line = ID()
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(stmt, line, info),
                    ad_stmt_info(line, stmt, nothing),
                )
            end
        end
    end

    interp = Taped.TInterp()
    @testset "$(_typeof((f, x...)))" for (n, (interface_only, perf_flag, bnds, f, x...)) in
        collect(enumerate(TestResources.generate_test_functions()))

        sig = _typeof((f, x...))
        @info "$n: $sig"
        TestUtils.test_derived_rule(
            Xoshiro(123456), f, x...; interp, perf_flag, interface_only, is_primitive=false
        )

        # codual_args = map(zero_codual, (f, x...))
        # rule = Taped.build_rrule(interp, sig)
        # out, pb!! = rule(codual_args...)
        # # @code_warntype optimize=true rule(codual_args...)
        # # @code_warntype optimize=true pb!!(tangent(out), map(tangent, codual_args)...)

        # primal_time = @benchmark $f($(Ref(x))[]...)
        # s2s_time = @benchmark $rule($codual_args...)[2]($(tangent(out)), $(map(tangent, codual_args))...)
        # in_f = in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp);
        # __rrule!! = Taped.build_rrule!!(in_f);
        # df = zero_codual(in_f);
        # codual_x = map(zero_codual, (f, x...));
        # interp_time = @benchmark TestUtils.to_benchmark($__rrule!!, $df, $codual_x...)

        # display(primal_time)
        # display(s2s_time)
        # display(interp_time)
        # s2s_ratio = time(s2s_time) / time(primal_time)
        # interp_ratio = time(interp_time) / time(primal_time)
        # println("s2s ratio ratio: $(s2s_ratio)")
        # println("interp ratio: $(interp_ratio)")

        # f(rule, codual_args, out) = rule(codual_args...)[2](tangent(out), map(tangent, codual_args)...)
        # f(rule, codual_args, out)
        # @profview(run_many_times(1_000, f, rule, codual_args, out))
    end
end
