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
end
