module IRUtilsGlobalRefs
    __x_1 = 5.0
    const __x_2 = 5.0
    __x_3::Float64 = 5.0
    const __x_4::Float64 = 5.0

    foo(x) = x
    foo(x::Float64) = 5x
end

@testset "ir_utils" begin
    @testset "ircode $(typeof(fargs))" for fargs in Any[
        (sin, 5.0), (cos, 1.0),
    ]
        # Construct a vector of instructions from known function.
        f, args... = fargs
        insts = only(code_typed(f, _typeof(args)))[1].code
    
        # Use Tapir.ircode to build an `IRCode`.
        argtypes = Any[map(_typeof, fargs)...]
        ir = Tapir.ircode(insts, argtypes)

        # Check the validity of the `IRCode`, and that an OpaqueClosure constructed using it
        # gives the same answer as the original function.
        @test length(ir.stmts.inst) == length(insts)
        @test Core.OpaqueClosure(ir; do_compile=true)(args...) == f(args...)
    end
    @testset "infer_ir!" begin

        # Generate IR without any types. 
        ir = Tapir.ircode(
            Any[
                Expr(:call, GlobalRef(Base, :sin), Argument(2)),
                Expr(:call, cos, SSAValue(1)),
                ReturnNode(SSAValue(2)),
            ],
            Any[Tuple{}, Float64],
        )

        # Run inference and check that the types are as expected.
        ir = Tapir.infer_ir!(ir)
        @test ir.stmts.type[1] == Float64
        @test ir.stmts.type[2] == Float64

        # Check that the ir is runable.
        @test Core.OpaqueClosure(ir)(5.0) == cos(sin(5.0))
    end
    @testset "unhandled_feature" begin
        @test_throws Tapir.UnhandledLanguageFeatureException Tapir.unhandled_feature("foo")
    end
    @testset "inc_args" begin
        @test Tapir.inc_args(Expr(:call, sin, Argument(4))) == Expr(:call, sin, Argument(5))
        @test Tapir.inc_args(ReturnNode(Argument(2))) == ReturnNode(Argument(3))
        id = ID()
        @test Tapir.inc_args(IDGotoIfNot(Argument(1), id)) == IDGotoIfNot(Argument(2), id)
        @test Tapir.inc_args(IDGotoNode(id)) == IDGotoNode(id)
    end
    @testset "lookup_invoke_ir" begin

        # Bail out if a non-invoke signature is passed in.
        @test_throws(
            ArgumentError,
            Tapir.lookup_invoke_ir(Tapir.TapirInterpreter(), Tuple{typeof(sin), Float64}),
        )

        ir = Tapir.lookup_invoke_ir(
            Tapir.TapirInterpreter(),
            Tuple{typeof(invoke), typeof(IRUtilsGlobalRefs.foo), Type{Tuple{Any}}, Float64},
        )
        oc = Core.OpaqueClosure(ir; do_compile=true)
        @test oc(5.0) == 5.0
        @test invoke(IRUtilsGlobalRefs.foo, Tuple{Any}, 5.0) == 5.0
    end
end
