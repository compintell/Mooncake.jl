@testset "unrolled_function" begin
    @testset "const_coinstruction" begin
        inst = const_coinstruction(CoDual(5.0, 4.0))
        @test isempty(input_primals(inst))
        @test isempty(input_tangents(inst))
        @test output_primal(inst) == 5.0
        @test output_tangent(inst) == 4.0
    end
    @testset "might_be_active" begin
        @test might_be_active(Float64)
        @test !might_be_active(Int)
        @test might_be_active(Vector{Float64})
        @test !might_be_active(Vector{Bool})
        @test !might_be_active(Tuple{Bool})
        @test !might_be_active(Tuple{Bool, Bool})
        @test might_be_active(Tuple{Bool, Float64})
        @test might_be_active(Tuple{Bool, Vector{Float64}})
        @test might_be_active(TestResources.Foo)
        @test might_be_active(TestResources.StructFoo)
        @test might_be_active(TestResources.MutableFoo)
    end
    @testset "build_construction and pullback! $f" for (perf_flag, f, args...) in
        TestResources.PRIMITIVE_TEST_FUNCTIONS

        # Specify input tangents.
        rng = Xoshiro(123456)
        dargs = map(Base.Fix1(randn_tangent, rng), args)
        original_args = map(deepcopy, args)
        original_dargs = map(deepcopy, dargs)

        # Construct coinstruction.
        dual_args = (CoDual(f, NoTangent()), map(CoDual, args, dargs)...)
        inputs = map(const_coinstruction, dual_args)
        inst = build_coinstruction(inputs...)
        @test all(map(==, input_tangents(inst), (NoTangent(), dargs...)))

        # Seed the output tangent and check that seeding has occured.
        dout = randn_tangent(rng, output_primal(inst))
        seed_output_tangent!(inst, copy(dout))
        @test output_tangent(inst) == dout

        # Run the reverse-pass of the coinstruction.
        pullback!(inst)

        # Run the test case without the co-instruction wrappers.
        other_dual_args = (
            CoDual(f, NoTangent()), map(CoDual, original_args, original_dargs)...,
        )
        _out, _pb!! = rrule!!(other_dual_args...)
        _out = set_tangent!!(_out, copy(dout))
        new_dargs = _pb!!(tangent(_out), NoTangent(), original_dargs...)

        # Check that the memory has been set up such that the data used with the
        # coinstruction does _not_ alias the data used with the `rrule!!`.
        @assert !ismutable(primal(_out)) || primal(_out) !== output_primal(inst)
        @assert !ismutable(tangent(_out)) || tangent(_out) !== output_tangent(inst)

        # Compare the result of running the coinstruction forwards- and backwards against
        # manually running the rrule!!. Should yield _exactly_ the same results.
        @test primal(_out) == output_primal(inst)
        @test tangent(_out) == output_tangent(inst)
        @test all(map(==, new_dargs, input_tangents(inst)))
    end
    @testset "$f, $typeof(x)" for (interface_only, f, x...) in TestResources.TEST_FUNCTIONS
        @info "$(map(typeof, (f, x...)))"
        test_taped_rrule!!(
            Xoshiro(123456), f, deepcopy(x)...; interface_only, perf_flag=:none
        )
    end
    @testset "acceleration $f" for (_, f, args...) in TestResources.TEST_FUNCTIONS

        x = (f, deepcopy(args)...)
        x_x̄ = map(CoDual, x, map(zero_tangent, x))
        x_x̄_copy = deepcopy(x_x̄)

        y, tape = Taped.trace(f, deepcopy(args)...; ctx=Taped.RMC())
        f_ur = Taped.UnrolledFunction(tape)

        ȳ = randn_tangent(Xoshiro(123456), y)
        ȳ_copy = deepcopy(ȳ)

        # Construct accelerated tape and use to compute gradients.
        fast_tape = Taped.construct_accel_tape(CoDual(f_ur, NoTangent()), x_x̄...)
        x̄ = Taped.execute!(fast_tape, ȳ, x_x̄...)

        # Use regular unrolled tape rrule to compute gradients.
        _, tape = Taped.trace(f, deepcopy(args)...; ctx=Taped.RMC())
        f_ur = Taped.UnrolledFunction(tape)
        y_ȳ, pb!! = Taped.rrule!!(CoDual(f_ur, NoTangent()), x_x̄_copy...)
        new_ȳ = increment!!(set_to_zero!!(tangent(y_ȳ)), ȳ_copy)
        x̄_std = pb!!(new_ȳ, NoTangent(), map(tangent, x_x̄_copy)...)

        # Check that the result agrees with standard execution.
        @test all(map(==, x̄, x̄_std))
    end
end
