@testset "reverse_mode_ad" begin
    @testset for (f, x) in TestResources.UNARY_FUNCTIONS[1:2]
        _, tape = Umlaut.trace(f, x; ctx=Taped.RMC())
        x_dx = Shadow(x, Ref(0.0), nothing)
        rm_tape = to_reverse_mode_ad(tape)
        play!(rm_tape, f, x_dx)
        @test ReverseDiff.gradient(x -> f(only(x)), [x])[1] ≈ Taped.shadow(x_dx)[]
    end
end

function performance_test()
    x = Shadow(5.0, Ref(0.0), nothing)
    y = Taped.rrule(sin, x)
    shadow(y)[] = 1.0

    # Check that pullback for simple operation is performant.
    display(@benchmark $y.pb!())
    println()

    # Check that pullback from inside funciton wrapper is performant.
    wrapper = FunctionWrapper{Nothing, Tuple{}}(Taped.ReverseExecutor(y))
    display(wrapper())
    println()

    display(@benchmark $wrapper())
    println()

    wrappers = Vector{FunctionWrapper{Nothing, Tuple{}}}(undef, 2)
    wrappers[1] = wrapper
    wrappers[2] = wrapper
end



# # val = f(args...)
# #
# # becomes
# #
# # args = dereference.(arg_refs)
# # val = f(args...)
# # val_ref[] = val

# struct Instruction{Tf, Targ_refs, Tval_ref}
#     f::Tf
#     arg_refs::Targ_refs
#     val_ref::Tval_ref
# end

# function (inst::Instruction)()
#     args = map(getindex, inst.arg_refs)
#     val = inst.f(args...)
#     inst.val_ref[] = val
# end

# inst = Instruction(sin, (Ref(5.0), ), Ref(0.0))
# inst()

# insts = FunctionWrapper{Nothing, Tuple{}}.(fill(inst, 3))
# function execute_instructions(insts)
#     for inst in insts
#         inst()
#     end
# end

# @benchmark execute_instructions($insts)
