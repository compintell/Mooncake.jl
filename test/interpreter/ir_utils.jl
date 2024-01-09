@testset "ir_utils" begin
    @testset "ircode $(typeof(fargs))" for fargs in Any[
        (sin, 5.0), (cos, 1.0),
    ]
        # Construct a vector of instructions from known function.
        f, args... = fargs
        insts = only(code_typed(f, Tuple{map(Core.Typeof, args)...}))[1].code
    
        # Use Taped.ircode to build an `IRCode`.
        argtypes = Any[map(Core.Typeof, fargs)...]
        ir = Taped.ircode(insts, argtypes)

        # Check the validity of the `IRCode`, and that an OpaqueClosure constructed using it
        # gives the same answer as the original function.
        @test length(ir.stmts.inst) == length(insts)
        @test Core.OpaqueClosure(ir; do_compile=true)(args...) == f(args...)
    end
    @testset "infer_ir!" begin

        # Generate IR without any types. 
        ir = Taped.ircode(
            Any[
                Expr(:call, GlobalRef(Base, :sin), Argument(2)),
                Expr(:call, cos, SSAValue(1)),
                ReturnNode(SSAValue(2)),
            ],
            Any[Tuple{}, Float64],
        )

        # Run inference and check that the types are as expected.
        ir = Taped.infer_ir!(ir)
        @test ir.stmts.type[1] == Float64
        @test ir.stmts.type[2] == Float64

        # Check that the ir is runable.
        @test Core.OpaqueClosure(ir)(5.0) == cos(sin(5.0))
    end
end
