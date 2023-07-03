rd_grad(f, x::Float64) = ReverseDiff.gradient(x -> f(only(x)), [x])[1]
function rd_grad(f, x::Array{Float64})
    return only(FiniteDifferences.grad(central_fdm(5, 1), f ∘ copy, x))
end

function test_ad(f, x)
    x_copy = deepcopy(x)
    dx, g = Taped.gradient(f, x)
    @test rd_grad(f, x_copy) ≈ dx
    @test x_copy ≈ x

    @test rd_grad(f, x_copy) ≈ g(f, x)
    @test x_copy ≈ x
end

get_address(x) = ismutable(x) ? pointer_from_objref(x) : nothing

function test_rrule!!(rng::AbstractRNG, x...)

    # Set up problem.
    x_copy = deepcopy(x)
    x_addresses = map(get_address, x)
    x_x̄ = map(x -> CoDual(x, randn_tangent(rng, x)), x)
    y_ȳ, pb!! = Taped.rrule!!(x_x̄...)
    x = map(primal, x_x̄)
    x̄ = map(shadow, x_x̄)

    # Check output and incremented shadow types are correct.
    @test typeof(primal(y_ȳ)) == typeof(x[1](x[2:end]...))
    @test primal(y_ȳ) == x[1](x[2:end]...)
    @test shadow(y_ȳ) isa tangent_type(typeof(primal(y_ȳ)))
    x̄_new = pb!!(shadow(y_ȳ), x̄...)
    @test all(map((a, b) -> typeof(a) == typeof(b), x̄_new, x̄))

    # Check aliasing.
    @test all(map((x̄, x̄_new) -> ismutable(x̄) ? x̄ === x̄_new : true, x̄, x̄_new))

    # Check that inputs have been returned to their original state.
    @test all(map(==, x, x_copy))

    # Check that memory addresses have remained constant.
    new_x_addresses = map(get_address, x)
    @test all(map(==, x_addresses, new_x_addresses))

    # Check that the answers are numerically correct.
    Taped.test_rmad(rng, x...)
end

@testset "reverse_mode_ad" begin
    @testset "$f, $(typeof(x))" for (f, x...) in [

        # IR-node workarounds:
        (Taped.Umlaut.__new__, UnitRange{Int}, 5, 9),
        (Taped.Umlaut.__new__, TestResources.StructFoo, 5.0, randn(4)),
        (Taped.Umlaut.__new__, TestResources.MutableFoo, 5.0, randn(5)),

        # Built-ins:
        (===, 5.0, 4.0),
        (===, 5.0, randn(5)),
        (===, randn(5), randn(3)),
        (===, 5.0, 5.0),
        (Core.apply_type, Vector, Float64),
        (Core.apply_type, Array, Float64, 2),
        (fieldtype, TestResources.StructFoo, :a),
        (fieldtype, TestResources.StructFoo, :b),
        (fieldtype, TestResources.MutableFoo, :a),
        (fieldtype, TestResources.MutableFoo, :b),
        # (getfield, TestResources.StructFoo(5.0), :a),
        (getfield, TestResources.StructFoo(5.0, randn(5)), :b),
        # (getfield, TestResources.MutableFoo(5.0), :a),
        (getfield, TestResources.MutableFoo(5.0, randn(5)), :b),
        (getfield, UnitRange{Int}(5:9), :start),
        (getfield, UnitRange{Int}(5:9), :stop),
        (setfield!, TestResources.MutableFoo(5.0, randn(5)), :a, 4.0),
        (setfield!, TestResources.MutableFoo(5.0, randn(5)), :b, randn(5)),
        (tuple, 5.0, 4.0),
        (tuple, randn(5), 5.0),
        (tuple, randn(5), randn(4)),
        (tuple, 5.0, randn(1)),
        (typeassert, 5.0, Float64),
        (typeassert, randn(5), Vector{Float64}),
        (typeof, 5.0),
        (typeof, randn(5)),

        # Non-essential rules:
        (sin, 5.0),
        (cos, 5.0),
        (getindex, randn(5), 4),
        (getindex, randn(5, 4), 1, 3),
        (setindex!, randn(5), 4.0, 3),
        (setindex!, randn(5, 4), 3.0, 1, 3),
    ]
        test_rrule!!(Xoshiro(123456), f, x...)
    end
    @testset for (f, x) in TestResources.UNARY_FUNCTIONS[1:6]
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
