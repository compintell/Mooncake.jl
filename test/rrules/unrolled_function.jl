function set_args!(acc_tape, args...)
    foreach(enumerate(args)) do (n, arg)
        current_arg = acc_tape.arg_refs[n][].output
        current_arg.output[] = arg
    end
    return nothing
end

function execute!(tape)
    for inst in tape.instructions
        inst()
    end
    return Taped.get_return_val(tape)
end

@testset "unrolled_function" begin
    @testset "const_coinstruction" begin
        inst = const_coinstruction(CoDual(5.0, 4.0))
        @test isempty(input_primals(inst))
        @test isempty(input_shadows(inst))
        @test output_primal(inst) == 5.0
        @test output_shadow(inst) == 4.0
    end
    @testset "build_construction and pullback! $f" for (f, args...) in
        TestResources.PRIMITIVE_TEST_FUNCTIONS

        # Specify input shadows.
        rng = Xoshiro(123456)
        dargs = map(Base.Fix1(randn_tangent, rng), args)
        original_args = map(deepcopy, args)
        original_dargs = map(deepcopy, dargs)

        # Construct coinstruction.
        dual_args = (CoDual(f, NoTangent()), map(CoDual, args, dargs)...)
        inputs = map(const_coinstruction, dual_args)
        inst = build_coinstruction(inputs...)
        @test all(map(==, input_shadows(inst), (NoTangent(), dargs...)))

        # Seed the output shadow and check that seeding has occured.
        dout = randn_tangent(rng, output_primal(inst))
        seed_output_shadow!(inst, copy(dout))
        @test output_shadow(inst) == dout

        # Run the reverse-pass of the coinstruction.
        pullback!(inst)

        # Run the test case without the co-instruction wrappers.
        other_dual_args = (
            CoDual(f, NoTangent()), map(CoDual, original_args, original_dargs)...,
        )
        _out, _pb!! = rrule!!(other_dual_args...)
        _out = set_shadow!!(_out, copy(dout))
        new_dargs = _pb!!(shadow(_out), NoTangent(), dargs...)

        # Check that the memory has been set up such that the data used with the
        # coinstruction does _not_ alias the data used with the `rrule!!`.
        @assert !ismutable(primal(_out)) || primal(_out) !== output_primal(inst)
        @assert !ismutable(shadow(_out)) || shadow(_out) !== output_shadow(inst)

        # Compare the result of running the coinstruction forwards- and backwards against
        # manually running the rrule!!. Should yield _exactly_ the same results.
        @test primal(_out) == output_primal(inst)
        @test shadow(_out) == output_shadow(inst)
        @test all(map(==, new_dargs, input_shadows(inst)))
    end
    @testset for (interface_only, f, x...) in vcat(
        TestResources.TEST_FUNCTIONS,
        [
            (false, getindex, randn(5), 4),
            (false, getindex, randn(5, 4), 1, 3),
            (false, setindex!, randn(5), 4.0, 3),
            (false, setindex!, randn(5, 4), 3.0, 1, 3),
            (false, x -> getglobal(Main, :sin)(x), 5.0),
            (false, x -> pointerref(bitcast(Ptr{Float64}, pointer_from_objref(Ref(x))), 1, 1), 5.0),
            (false, (v, x) -> (pointerset(pointer(x), v, 2, 1); x), 3.0, randn(5)),
            (false, x -> (pointerset(pointer(x), UInt8(3), 2, 1); x), rand(UInt8, 5)),
            (false, x -> Ref(x)[], 5.0),
            (false, x -> unsafe_load(bitcast(Ptr{Float64}, pointer_from_objref(Ref(x)))), 5.0),
            (false, x -> unsafe_load(Base.unsafe_convert(Ptr{Float64}, x)), randn(5)),
            (false, view, randn(5, 4), 1, 1),
            (false, view, randn(5, 4), 2:3, 1),
            (false, view, randn(5, 4), 1, 2:3),
            (false, view, randn(5, 4), 2:3, 2:4),
            (true, Array{Float64, 1}, undef, (1, )),
            (true, Array{Float64, 2}, undef, (2, 3)),
            (true, Array{Float64, 3}, undef, (2, 3, 4)),
            (false, Xoshiro, 123456),
            (false, push!, randn(5), 3.0),
        ],
        map(n -> (false, map, sin, (randn(n)..., )), 1:7),
        map(n -> (false, map, sin, randn(n)), 1:7),
        map(n -> (false, x -> sin.(x), (randn(n)..., )), 1:7),
        map(n -> (false, x -> sin.(x), randn(n)), 1:7),
        vec(map(Iterators.product( # These all work fine, but take a long time to run.
            [randn(3, 5), transpose(randn(5, 3)), adjoint(randn(5, 3))],
            [
                randn(3, 4),
                transpose(randn(4, 3)),
                adjoint(randn(4, 3)),
                view(randn(5, 5), 1:3, 1:4),
                transpose(view(randn(5, 5), 1:4, 1:3)),
                adjoint(view(randn(5, 5), 1:4, 1:3)),
            ],
            [
                randn(4, 5),
                transpose(randn(5, 4)),
                adjoint(randn(5, 4)),
                view(randn(5, 5), 1:4, 1:5),
                transpose(view(randn(5, 5), 1:5, 1:4)),
                adjoint(view(randn(5, 5), 1:5, 1:4)),
            ],
        )) do (A, B, C)
            (false, mul!, A, B, C, randn(), randn())
        end),
        TestResources.DIFFTESTS_FUNCTIONS,
    )
        @info "$(map(typeof, (f, x...)))"
        test_taped_rrule!!(Xoshiro(123456), f, map(deepcopy, x)...; interface_only)
    end
    @testset "acceleration" begin
        f = TestResources.test_mlp
        args = (randn(5, 2), randn(7, 5), randn(3, 7))
        y, tape = Taped.trace(f, args...; ctx=Taped.RMC())
        f_ur = Taped.UnrolledFunction(tape)
        x = (f, args...)
        dx = map(zero_tangent, x)

        fast_tape = Taped.construct_accel_tape(
            zero_tangent(y), CoDual(f_ur, NoTangent()), map(CoDual, x, dx)...
        )
        execute!(fast_tape)
    end
end
