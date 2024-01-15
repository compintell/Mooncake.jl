@testset "abstract_interpretation" begin
    # Check that inlining doesn't / does happen as expected.
    @testset "TapedInterpreter" begin
        @testset "non-primitive continues to be inlined away" begin

            # A non-primitive is present in the IR for contains_non_primitive. It is
            # inlined away under usual interpretation, and should also be inlined away
            # when doing AD.
            sig = Tuple{typeof(Taped.contains_non_primitive), Float64}

            # Pre-condition: must inline away under usual compilation.
            usual_ir = Base.code_ircode_by_type(sig)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), usual_ir.stmts.inst)
            @assert usual_ir.stmts.inst[invoke_line].args[2] == GlobalRef(Taped, :sin)

            # Should continue to inline away under AD compilation.
            interp = Taped.TapedInterpreter(Taped.DefaultCtx())
            ad_ir = Base.code_ircode_by_type(sig; interp)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), ad_ir.stmts.inst)
            @test ad_ir.stmts.inst[invoke_line].args[2] == GlobalRef(Taped, :sin)
        end
        @testset "primitive is no longer inlined away" begin

            # A primitive is present in the IR for contains_primitive. It is inlined away
            # under usual interpretation, but should not be when doing AD.
            sig = Tuple{typeof(Taped.contains_primitive), Float64}

            # Pre-condition: must inline away under usual compilation.
            usual_ir = Base.code_ircode_by_type(sig)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), usual_ir.stmts.inst)
            @assert usual_ir.stmts.inst[invoke_line].args[2] == GlobalRef(Taped, :sin)

            # Should not inline away under AD compilation.
            interp = Taped.TapedInterpreter(Taped.DefaultCtx())
            ad_ir = Base.code_ircode_by_type(sig; interp)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), ad_ir.stmts.inst)
            @test ad_ir.stmts.inst[invoke_line].args[2] == GlobalRef(Taped, :a_primitive)
        end
        @testset "deep primitive is not inlined away" begin

            # A non-primitive is immediately visible in the IR, but this non-primitive is
            # usually inlined away to reveal a primitive. This primitive is _also_ usually
            # inlined away, but should not be when doing AD. This case is not handled if
            # various bits of information are not properly propagated in the compiler.
            sig = Tuple{typeof(Taped.contains_primitive_behind_call), Float64}

            # Pre-condition: both functions should be inlined away under usual conditions.
            usual_ir = Base.code_ircode_by_type(sig)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), usual_ir.stmts.inst)
            @assert usual_ir.stmts.inst[invoke_line].args[2] == GlobalRef(Taped, :sin)

            # Should not inline away under AD compilation.
            interp = Taped.TapedInterpreter(Taped.DefaultCtx())
            ad_ir = Base.code_ircode_by_type(sig; interp)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), ad_ir.stmts.inst)
            @test ad_ir.stmts.inst[invoke_line].args[2] == GlobalRef(Taped, :a_primitive)
        end
    end
end