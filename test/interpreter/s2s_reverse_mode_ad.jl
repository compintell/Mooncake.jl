module S2SGlobals
    using LinearAlgebra, Mooncake

    non_const_global = 5.0
    const const_float = 5.0
    const const_int = 5
    const const_bool = true

    # used for regression test for issue 184
    struct A
        data
    end
    f(a, x) = dot(a.data, x)

    # Test cases designed to cause `LazyDerivedRule` to throw an error when attempting to
    # construct a rule for `bar`.
    foo(x) = x
    @noinline bar(x) = foo(x)
    baz(x) = bar(x)
    Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{typeof(foo), Any}
end

@testset "s2s_reverse_mode_ad" begin
    @testset "SharedDataPairs" begin
        m = SharedDataPairs()
        id = Mooncake.add_data!(m, 5.0)
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
        is_used_dict = Dict{ID, Bool}(id_ssa_1 => true, id_ssa_2 => true)
        rdata_ref = Ref{Tuple{map(Mooncake.lazy_zero_rdata_type, (Float64, Int))...}}()
        info = ADInfo(get_interpreter(), arg_types, ssa_insts, is_used_dict, false, rdata_ref)

        # Verify that we can access the interpreter and terminator block ID.
        @test info.interp isa Mooncake.MooncakeInterpreter

        # Verify that we can get the type associated to Arguments, IDs, and others.
        global ___x = 5.0
        global ___y::Float64 = 5.0
        @test Mooncake.get_primal_type(info, Argument(1)) == Float64
        @test Mooncake.get_primal_type(info, Argument(2)) == Int
        @test Mooncake.get_primal_type(info, id_ssa_1) == Float64
        @test Mooncake.get_primal_type(info, GlobalRef(Base, :sin)) == typeof(sin)
        @test Mooncake.get_primal_type(info, GlobalRef(Main, :___x)) == Any
        @test Mooncake.get_primal_type(info, GlobalRef(Main, :___y)) == Float64
        @test Mooncake.get_primal_type(info, 5) == Int
        @test Mooncake.get_primal_type(info, QuoteNode(:hello)) == Symbol
    end
    @testset "ADStmtInfo" begin
        # If the ID passes as the comms channel doesn't appear in the stmts for the forwards
        # pass, then this constructor ought to error.
        @test_throws ArgumentError ad_stmt_info(ID(), ID(), nothing, nothing)
    end
    @testset "make_ad_stmts!" begin

        # Set up ADInfo -- this state is required by `make_ad_stmts!`, and the
        # `LineToADDataMap` object can be mutated.
        id_line_1 = ID()
        id_line_2 = ID()
        info = ADInfo(
            get_interpreter(),
            Dict{Argument, Any}(Argument(1) => typeof(sin), Argument(2) => Float64),
            Dict{ID, CC.NewInstruction}(
                id_line_1 => new_inst(Expr(:invoke, nothing, cos, Argument(2)), Float64),
                id_line_2 => new_inst(nothing, Any),
            ),
            Dict{ID, Bool}(id_line_1=>true, id_line_2=>true),
            false,
            Ref{Tuple{map(Mooncake.lazy_zero_rdata_type, (typeof(sin), Float64))...}}(),
        )

        @testset "Nothing" begin
            line = ID()
            @test TestUtils.has_equal_data(
                make_ad_stmts!(nothing, line, info),
                ad_stmt_info(line, nothing, nothing, nothing),
            )
        end
        @testset "ReturnNode" begin
            line = ID()
            @testset "unreachable" begin
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(ReturnNode(), line, info),
                    ad_stmt_info(line, nothing, ReturnNode(), nothing),
                )
            end
            @testset "Argument" begin
                val = Argument(4)
                stmts = make_ad_stmts!(ReturnNode(Argument(2)), line, info)
                @test only(stmts.fwds)[2].stmt == ReturnNode(Argument(3))
                @test Meta.isexpr(only(stmts.rvs)[2].stmt, :call)
                @test only(stmts.rvs)[2].stmt.args[1] == Mooncake.increment_ref!
            end
            @testset "literal" begin
                stmt_info = make_ad_stmts!(ReturnNode(5.0), line, info)
                @test stmt_info isa ADStmtInfo
                @test stmt_info.fwds[2][2].stmt isa ReturnNode
            end
            @testset "GlobalRef" begin
                node = ReturnNode(GlobalRef(S2SGlobals, :const_float))
                stmt_info = make_ad_stmts!(node, line, info)
                @test stmt_info isa ADStmtInfo
                @test stmt_info.fwds[2][2].stmt isa ReturnNode
            end
        end
        @testset "IDGotoNode" begin
            line = ID()
            stmt = IDGotoNode(ID())
            @test TestUtils.has_equal_data(
                make_ad_stmts!(stmt, line, info), ad_stmt_info(line, nothing, stmt, nothing)
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
            stmt = IDPhiNode(ID[ID(), ID()], Any[ID(), 5.0])
            ad_stmts = make_ad_stmts!(stmt, id_line_1, info)
            @test ad_stmts isa ADStmtInfo
        end
        @testset "PiNode" begin
            @testset "π (nothing, Union{})" begin
                # This is a weird edge case that appeared in 1.11. See comment in src.
                line = id_line_1
                stmt_info = make_ad_stmts!(PiNode(nothing, Union{}), line, info)
                @test stmt_info isa ADStmtInfo
                @test last(stmt_info.fwds)[1] == line
            end
            @testset "π (nothing, Nothing)" begin
                stmt_info = make_ad_stmts!(PiNode(nothing, Nothing), id_line_1, info)
                @test stmt_info isa ADStmtInfo
                @test last(stmt_info.fwds)[1] == id_line_1
                fwds_stmt = last(stmt_info.fwds)[2].stmt
                @test fwds_stmt isa PiNode
                @test fwds_stmt.typ == CoDual{Nothing, NoFData}
                @test only(stmt_info.rvs)[2].stmt === nothing
            end
            @testset "π (nothing, CC.Const(nothing))" begin
                node = PiNode(nothing, CC.Const(nothing))
                stmt_info = make_ad_stmts!(node, id_line_1, info)
                @test stmt_info isa ADStmtInfo
                @test last(stmt_info.fwds)[1] == id_line_1
                fwds_stmt = last(stmt_info.fwds)[2].stmt
                @test fwds_stmt isa PiNode
                @test fwds_stmt.typ == CoDual{Nothing, NoFData}
                @test only(stmt_info.rvs)[2].stmt === nothing
            end
            @testset "π (GlobalRef, Type)" begin
                node = PiNode(GlobalRef(S2SGlobals, :const_float), Any)
                stmt_info = make_ad_stmts!(node, id_line_1, info)
                @test stmt_info isa ADStmtInfo
                fwds_stmt = last(stmt_info.fwds)[2].stmt
                @test fwds_stmt isa PiNode
                @test fwds_stmt.typ == CoDual
                @test only(stmt_info.rvs)[2].stmt === nothing
            end
            @testset "sharpen type of ID" begin
                line = id_line_1
                val = id_line_2
                stmt_info = make_ad_stmts!(PiNode(val, Float64), line, info)
                @test stmt_info isa ADStmtInfo
            end
        end
        @testset "GlobalRef" begin
            @testset "non-const" begin
                global_ref = GlobalRef(S2SGlobals, :non_const_global)
                stmt_info = make_ad_stmts!(global_ref, ID(), info)
                @test Mooncake.TestResources.non_const_global_ref(5.0) == 5.0 # run primal
                @test stmt_info isa Mooncake.ADStmtInfo
                @test Meta.isexpr(last(stmt_info.fwds)[2].stmt, :call)
                @test last(stmt_info.fwds)[2].stmt.args[1] == Mooncake.__verify_const
            end
            @testset "differentiable const globals" begin
                stmt_info = make_ad_stmts!(GlobalRef(S2SGlobals, :const_float), ID(), info)
                @test stmt_info isa Mooncake.ADStmtInfo
                @test only(stmt_info.fwds)[2].stmt isa Expr
                @test only(stmt_info.fwds)[2].stmt.args[1] === Mooncake.uninit_fcodual
            end
        end
        @testset "PhiCNode" begin
            @test_throws(
                Mooncake.UnhandledLanguageFeatureException,
                make_ad_stmts!(Core.PhiCNode(Any[]), ID(), info),
            )
        end
        @testset "UpsilonNode" begin
            @test_throws(
                Mooncake.UnhandledLanguageFeatureException,
                make_ad_stmts!(Core.UpsilonNode(5), ID(), info),
            )
        end
        @testset "Expr" begin
            @testset "assignment to GlobalRef" begin
                @test_throws(
                    Mooncake.UnhandledLanguageFeatureException,
                    make_ad_stmts!(Expr(:(=), GlobalRef(Main, :a), 5.0), ID(), info)
                )
            end
            @testset "copyast" begin
                stmt = Expr(:copyast, QuoteNode(:(hi)))
                ad_stmts = make_ad_stmts!(stmt, ID(), info)
                @test ad_stmts isa Mooncake.ADStmtInfo
                @test Meta.isexpr(ad_stmts.fwds[1][2].stmt, :call)
                @test ad_stmts.fwds[1][2].stmt.args[1] == identity
            end
            @testset "throw_undef_if_not" begin
                cond_id = ID()
                line = ID()
                fwds = Expr(:throw_undef_if_not, :x, cond_id)
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(Expr(:throw_undef_if_not, :x, cond_id), line, info),
                    ad_stmt_info(line, nothing, fwds, nothing),
                )
            end
            @testset "$stmt" for stmt in [
                Expr(:gc_preserve_begin),
            ]
                line = ID()
                @test TestUtils.has_equal_data(
                    make_ad_stmts!(stmt, line, info),
                    ad_stmt_info(line, nothing, stmt, nothing),
                )
            end
        end
    end
    @testset "rule_type $sig, $debug_mode" for
        sig in Any[
            Tuple{typeof(getfield), Tuple{Float64}, 1},
            Tuple{typeof(TestResources.foo), Float64},
            Tuple{typeof(TestResources.type_unstable_tester_0), Ref{Any}},
            Tuple{typeof(TestResources.tuple_with_union), Bool},
        ],
        debug_mode in [true, false]

        interp = get_interpreter()
        rule = Mooncake.build_rrule(interp, sig; debug_mode)
        @test rule isa Mooncake.rule_type(interp, sig; debug_mode)
    end
    @testset "LazyDerivedRule" begin
        fargs = (S2SGlobals.baz, 5.0)
        rule = build_rrule(fargs...)
        msg = "Unable to put rule in rule field. A `BadRuleTypeException` might be thrown."
        @test_logs(
            (:warn, msg),
            (@test_throws Mooncake.BadRuleTypeException rule(map(zero_fcodual, fargs)...)),
        )
    end
    @testset "MooncakeRuleCompilationError" begin
        @test_throws(Mooncake.MooncakeRuleCompilationError, Mooncake.build_rrule(sin))
    end
    @testset "$(_typeof((f, x...)))" for (n, (interface_only, perf_flag, bnds, f, x...)) in
        collect(enumerate(TestResources.generate_test_functions()))

        sig = _typeof((f, x...))
        @info "$n: $sig"
        TestUtils.test_rule(
            Xoshiro(123456), f, x...; perf_flag, interface_only, is_primitive=false
        )
        TestUtils.test_rule(
            Xoshiro(123456), f, x...;
            perf_flag=:none, interface_only, is_primitive=false, debug_mode=true,
        )

        # interp = Mooncake.get_interpreter()
        # codual_args = map(zero_codual, (f, x...))
        # fwds_args = map(Mooncake.to_fwds, codual_args)
        # rule = Mooncake.build_rrule(interp, sig)
        # out, pb!! = rule(fwds_args...)
        # # @code_warntype optimize=true rule(codual_args...)
        # # @code_warntype optimize=true pb!!(tangent(out), map(tangent, codual_args)...)

        # primal_time = @benchmark $f($(Ref(x))[]...)
        # s2s_time = @benchmark $rule($fwds_args...)[2]($(Mooncake.zero_rdata(primal(out))))

        # display(primal_time)
        # display(s2s_time)
        # s2s_ratio = time(s2s_time) / time(primal_time)
        # println("s2s ratio ratio: $(s2s_ratio)")

        # f(rule, fwds_args, out) = rule(fwds_args...)[2]((Mooncake.zero_rdata(primal(out))))
        # f(rule, fwds_args, out)
        # @profview(run_many_times(500, f, rule, fwds_args, out))
    end

    @testset "integration testing for invalid global ref errors" begin
        @test_throws(
            Mooncake.Mooncake.MooncakeRuleCompilationError,
            Mooncake.build_rrule(
                Tuple{typeof(Mooncake.TestResources.non_const_global_ref), Float64},
            )
        )
    end

    # Tests designed to prevent accidentally re-introducing issues which we have fixed.
    @testset "regression tests" begin

        # 184
        TestUtils.test_rule(
            Xoshiro(123456), S2SGlobals.f, S2SGlobals.A(2 * ones(3)), ones(3);
            interface_only=false, is_primitive=false,
        )

        # BenchmarkTools not working due to world age problems. Provided that this code
        # runs successfully, everything is okay -- no need to check anything specific.
        f(x) = sin(cos(x))
        rule = Mooncake.build_rrule(f, 0.0)
        @benchmark Mooncake.value_and_gradient!!($rule, $f, $(Ref(0.0))[])
    end
    @testset "literal Strings do not appear in shared data" begin
        f() = "hello"
        @test length(build_rrule(Tuple{typeof(f)}).fwds_oc.oc.captures) == 2
    end
    @testset "Literal Types do not appear in shared data" begin
        f() = Float64
        @test length(build_rrule(Tuple{typeof(f)}).fwds_oc.oc.captures) == 2
    end
end
