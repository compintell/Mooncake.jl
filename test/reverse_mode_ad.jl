rd_grad(f, x::Float64) = ReverseDiff.gradient(x -> f(only(x)), [x])[1]
function rd_grad(f, x::Array{Float64})
    return only(FiniteDifferences.grad(central_fdm(5, 1), f ∘ copy, x))
end

function test_ad(f, x)
    x_copy = copy(x)
    dx, g = Taped.gradient(f, x)
    @test rd_grad(f, x_copy) ≈ dx
    @test x_copy ≈ x

    @test rd_grad(f, x_copy) ≈ g(f, x)
    @test x_copy ≈ x
end

@testset "reverse_mode_ad" begin
    @testset for (f, x) in TestResources.UNARY_FUNCTIONS[1:7]
        test_ad(f, x)
    end
end

function performance_test()
    x = Shadow(5.0, Ref(0.0))
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
