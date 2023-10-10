function toy_tape()
    x_ref = Ref(5.0)
    y_ref = Ref(5.0)
    z_ref = Ref(5.0)
    inst_1 = Taped.Executor(Taped.Instruction(sin, (x_ref, ), y_ref))
    inst_2 = Taped.Executor(Taped.Instruction(cos, (y_ref, ), z_ref))
    arg_refs = (x_ref, )
    val_ref = z_ref
    return Taped.AcceleratedTape(
        FunctionWrapper{Nothing, Tuple{}}.([inst_1, inst_2]), arg_refs, val_ref
    )
end

# A simple function
function foo(x)
    y = sin(cos(x))
    y = cos(y)
    y = cos(y)
    y = cos(y)
    y = cos(y)
    y = cos(y)
    y = cos(y)
    return y
end

@testset "acceleration" begin
    tape = toy_tape()
    x = 5.0
    b_result = (@benchmark Taped.execute!($tape, $x))
    @test allocs(b_result) == 0
end
