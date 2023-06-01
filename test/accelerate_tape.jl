function toy_tape()
    x_ref = Ref(5.0)
    y_ref = Ref(5.0)
    z_ref = Ref(5.0)
    inst_1 = Taped.Executor(Instruction(sin, (x_ref, ), y_ref))
    inst_2 = Taped.Executor(Instruction(cos, (y_ref, ), z_ref))
    arg_refs = (x_ref, )
    val_ref = z_ref
    return AcceleratedTape(
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

@testset "acclerate_tape" begin
    tape = toy_tape()
    x = 5.0
    b_result = (@benchmark execute!($tape, $x))
    @test allocs(b_result) == 0

    _, tape = Umlaut.trace(foo, 5.0; ctx=Taped.RMC())
    r_tape = Taped.to_reverse_mode_ad(tape)
    x = Shadow(5.0, Ref(0.0), nothing)
    play!(r_tape, foo, x)
    acc_r_tape = Taped.accelerate(r_tape)
    b_result = (@benchmark execute!($acc_r_tape, foo, $x))
    @test allocs(b_result) == 3
end
