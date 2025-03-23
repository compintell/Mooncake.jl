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
        struct UserDefinedStruct
            a::Int64
            b::Vector{Float64}
            c::Vector{Vector{Float64}}
        end

        mutable struct UserDefinedMutableStruct
            a::Int64
            b::Vector{Float64}
            c::Vector{Vector{Float64}}
        end

        test_to_pass_cases = [
            (1, (1.0, 1.0)),
            (1.0, 1.0),
            (1, [[1.0, 1, 1.0], 1.0]),
            (1.0, [1.0]),
            UserDefinedStruct(1, [1.0, 1.0, 1.0], [[1.0]]),
            UserDefinedMutableStruct(1, [1.0, 1.0, 1.0], [[1.0]]),
            Dict(:a => [1, 2], :b => [3, 4]),
            Set([1, 2]),
        ]
        VERSION >= v"1.11" &&
            push!(test_to_pass_cases, fill!(Memory{Float64}(undef, 3), 3.0))

        @testset "Valid Output types" for res in test_to_pass_cases
            @test isnothing(Mooncake.__exclude_unsupported_output(res))
        end

        # Test when function outputs an invalid type. 
        test_to_fail_cases = []

        # ---- Aliasing Cases ----
        alias_vector = [[1, 2], [3, 4]]
        alias_vector[2] = alias_vector[1]
        push!(test_to_fail_cases, alias_vector)

        alias_tuple = ((1.0, 2.0), (3.0, 4.0))
        alias_tuple = (alias_tuple[1], alias_tuple[1])
        push!(test_to_fail_cases, alias_tuple)

        # ---- Circular Referencing Cases ----
        circular_vector = Any[[1.0, 2.0]]
        push!(circular_vector, circular_vector)
        push!(test_to_fail_cases, circular_vector)

        mutable struct CircularStruct
            data::Any
            numeric::Int64
        end

        circ_obj = CircularStruct(nothing, 1)
        circ_obj.data = circ_obj  # Self-referential struct
        push!(test_to_fail_cases, circ_obj)

        # ---- Unsupported Types ----
        push!(test_to_fail_cases, Ptr{Float64}(12345))
        push!(test_to_fail_cases, (1, Ptr{Float64}(12345)))
        push!(test_to_fail_cases, [Ptr{Float64}(12345)])

        @test_throws(
            Mooncake.ValueAndPullbackReturnTypeError,
            Mooncake.__exclude_unsupported_output.(test_to_fail_cases)
        )

        additional_test_set = Mooncake.tangent_test_cases()

        @testset for i in eachindex(additional_test_set)
            try
                Mooncake.__exclude_unsupported_output(additional_test_set[i][2])
            catch err
                @test isa(err, Mooncake.ValueAndPullbackReturnTypeError)
            end
        end
        additional_test_set = Mooncake.tangent_test_cases()
        @testset for i in eachindex(additional_test_set)
            original = additional_test_set[i][2]
            try
                if isnothing(Mooncake.__exclude_unsupported_output(original))
                    test_copy = Mooncake._copy_temp(original)
                    if typeof(test_copy) <: Mooncake._BuiltinArrays
                        for i in eachindex(test_copy)
                            if !isassigned(test_copy, i)
                                @test !isassigned(original, i)
                            else
                                @test test_copy[i] == original[i]
                            end
                        end

                        # isbitstypes with same values are stored in the same address (Value Caching).
                        if !isbitstype(typeof(original))
                            @test test_copy !== original
                        end
                    else
                        fields_copy = [
                            if !isdefined(test_copy, name)
                                nothing
                            else
                                getfield(test_copy, name)
                            end for name in fieldnames(typeof(test_copy))
                        ]
                        fields_orig = [
                            !isdefined(original, name) ? nothing : getfield(original, name)
                            for name in fieldnames(typeof(original))
                        ]
                        @test fields_copy == fields_orig

                        # Value caching for pure immutable Types!
                        if !any(isbitstype.(typeof.(fields_orig)))
                            @test !any(fields_copy .=== fields_orig)
                        end
                        @test typeof(test_copy) == typeof(original)
                    end
                end
            catch err
                @test isa(err, Mooncake.ValueAndPullbackReturnTypeError)
            end
        end
    end
end
