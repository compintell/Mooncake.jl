function count_allocs(fargs::P) where {P<:Tuple}
    f, args... = fargs
    f(args...) # warmup
    return @allocations f(args...)
end

@testset "interface" begin
    @testset "$(typeof((f, x...)))" for (ȳ, f, x...) in Any[
        (1.0, (x, y) -> x * y + sin(x) * cos(y), 5.0, 4.0),
        ([1.0, 1.0], x -> [sin(x), sin(2x)], 3.0),
        (1.0, x -> sum(5x), [5.0, 2.0]),
    ]
        @testset "debug_mode=$debug_mode" for debug_mode in Bool[false, true]
            rule = build_rrule(f, x...; debug_mode)
            v, (df, dx...) = value_and_pullback!!(rule, ȳ, f, x...)
            @test v ≈ f(x...)
            @test df isa tangent_type(typeof(f))
            for (_dx, _x) in zip(dx, x)
                @test _dx isa tangent_type(typeof(_x))
            end
        end
    end
    @testset "sensible error when CoDuals are passed to `value_and_pullback!!" begin
        foo(x) = sin(cos(x))
        rule = build_rrule(foo, 5.0)
        @test_throws ArgumentError value_and_pullback!!(rule, 1.0, foo, CoDual(5.0, 0.0))
    end
    @testset "value_and_gradient!!" begin
        @testset "($(typeof(fargs))" for fargs in Any[
            (sin, randn(Float64)),
            (sin, randn(Float32)),
            (x -> sin(cos(x)), randn(Float64)),
            (x -> sin(cos(x)), randn(Float32)),
            ((x, y) -> x + sin(y), randn(Float64), randn(Float64)),
            ((x, y) -> x + sin(y), randn(Float32), randn(Float32)),
            ((x...) -> x[1] + x[2], randn(Float64), randn(Float64)),
            (sum, randn(10)),
        ]
            kwargs = (debug_mode=false, silence_debug_messages=true)
            rule = build_rrule(fargs...; kwargs...)
            f, args... = fargs
            v, dfargs = value_and_gradient!!(rule, fargs...)
            @test v == f(args...)
            for (arg, darg) in zip(fargs, dfargs)
                @test tangent_type(typeof(arg)) == typeof(darg)
            end

            cache = Mooncake.prepare_gradient_cache(fargs...; kwargs...)
            _v, _dfargs = value_and_gradient!!(cache, fargs...)
            @test _v == v
            for (arg, darg) in zip(fargs, _dfargs)
                @test tangent_type(typeof(arg)) == typeof(darg)
            end
            alloc_count = count_allocs((value_and_gradient!!, cache, fargs...))
            if alloc_count > 0
                @test_broken alloc_count == 0
            else
                @test alloc_count == 0
            end
        end

        rule = build_rrule(identity, (5.0, 4.0))
        @test_throws(
            Mooncake.ValueAndGradientReturnTypeError,
            value_and_gradient!!(rule, identity, (5.0, 4.0)),
        )
        @test_throws(
            Mooncake.ValueAndGradientReturnTypeError,
            Mooncake.prepare_gradient_cache(identity, (5.0, 4.0)),
        )
    end
    @testset "value_and_pullback!!" begin
        @testset "($(typeof(fargs))" for (ȳ, fargs...) in Any[
            (randn(10), identity, randn(10)),
            (randn(), sin, randn(Float64)),
            (randn(), sum, randn(Float64)),
        ]
            kwargs = (debug_mode=false, silence_debug_messages=true)
            rule = build_rrule(fargs...; kwargs...)
            f, args... = fargs
            v, dfargs = value_and_pullback!!(rule, ȳ, fargs...)
            @test v == f(args...)
            for (arg, darg) in zip(fargs, dfargs)
                @test tangent_type(typeof(arg)) == typeof(darg)
            end

            cache = Mooncake.prepare_pullback_cache(fargs...; kwargs...)
            _v, _dfargs = value_and_pullback!!(cache, ȳ, fargs...)
            @test _v == v
            for (arg, darg) in zip(fargs, _dfargs)
                @test tangent_type(typeof(arg)) == typeof(darg)
            end
            alloc_count = count_allocs((value_and_pullback!!, cache, ȳ, fargs...))
            if alloc_count > 0
                @test_broken alloc_count == 0
            else
                @test alloc_count == 0
            end
        end
    end

    @testset "prepare_pullback_cache errors" begin
        # Test when function outputs a valid type.
        struct userdefinedstruct
            a::Int64
            b::Vector{Float64}
            c::Vector{Vector{Float64}}
        end

        mutable struct userdefinedmutablestruct
            a::Int64
            b::Vector{Float64}
            c::Vector{Vector{Float64}}
        end

        test_topass_cases = [
            (1, (1.0, 1.0)),
            (1.0, 1.0),
            (1, [[1.0, 1, 1.0], 1.0]),
            (1.0, [1.0]),
            userdefinedstruct(1, [1.0, 1.0, 1.0], [[1.0]]),
            userdefinedmutablestruct(1, [1.0, 1.0, 1.0], [[1.0]]),
        ]

        @testset "Valid Output types" for res in test_topass_cases
            @test isnothing(Mooncake.__exclude_unsupported_output(res))
        end

        # Test when function outputs an invalid type. 
        test_tofail_cases = []

        # ---- Aliasing Cases ----
        alias_vector = [[1, 2], [3, 4]]
        alias_vector[2] = alias_vector[1]
        push!(test_tofail_cases, alias_vector)

        alias_tuple = ((1.0, 2.0), (3.0, 4.0))
        alias_tuple = (alias_tuple[1], alias_tuple[1])
        push!(test_tofail_cases, alias_tuple)

        # ---- Circular Referencing Cases ----
        circular_vector = Any[[1.0, 2.0]]
        push!(circular_vector, circular_vector)
        push!(test_tofail_cases, circular_vector)
        mutable struct CircularStruct
            data::Any
            numeric::Int64
        end

        circ_obj = CircularStruct(nothing, 1)
        circ_obj.data = circ_obj  # Self-referential struct
        push!(test_tofail_cases, circ_obj)

        # ---- Unsupported Types ----
        push!(test_tofail_cases, Ptr{Float64}(12345))
        push!(test_tofail_cases, (1, Ptr{Float64}(12345)))
        push!(test_tofail_cases, [Ptr{Float64}(12345)])

        # ---- Non Differentiable Cases ----
        nondiff_dict = Dict(:a => [1, 2], :b => [3, 4])
        push!(test_tofail_cases, nondiff_dict)
        nondiff_set = Set([1, 2])
        push!(test_tofail_cases, nondiff_set)
        # push!(test_tofail_cases,Ptr{Float64}(1))

        @test_throws(
            Mooncake.ValueAndGradientReturnTypeError,
            Mooncake.__exclude_unsupported_output.(test_tofail_cases)
        )

        additional_testset = Mooncake.tangent_test_cases()

        for i in eachindex(additional_testset)
            try
                Mooncake.__exclude_unsupported_output(more[i][2])
            catch err
                @test isa(err, Mooncake.ValueAndGradientReturnTypeError)
            end
        end
    end
end
