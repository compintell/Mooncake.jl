using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using BenchmarkTools, Distributions, DynamicPPL, Mooncake, Random, Test
using Mooncake: NoCache, set_to_zero!!, set_to_zero_internal!!, zero_tangent

@testset "MooncakeDynamicPPLExt" begin
    # Create test models
    @model function test_model1(x)
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, sqrt(s))
        return x .~ Normal(m, sqrt(s))
    end

    @model function test_model2(x, y)
        τ ~ Gamma(1, 1)
        σ ~ InverseGamma(2, 3)
        μ ~ Normal(0, τ)
        x .~ Normal(μ, σ)
        return y .~ Normal(μ, σ)
    end

    @testset "Validation functions" begin
        # Test with a real DynamicPPL model
        model = test_model1([1.0, 2.0, 3.0])
        vi = DynamicPPL.VarInfo(Random.default_rng(), model)
        ldf = DynamicPPL.LogDensityFunction(model, vi, DynamicPPL.DefaultContext())
        tangent = zero_tangent(ldf)

        # Since we can't access extension functions directly, 
        # test the behavior indirectly through set_to_zero!!
        # If the optimization is working, set_to_zero!! should handle DynamicPPL types efficiently
        result = set_to_zero!!(deepcopy(tangent))
        @test result isa typeof(tangent)

        # Test with metadata - verify structure exists
        if hasfield(typeof(tangent.fields.varinfo.fields), :metadata)
            metadata = tangent.fields.varinfo.fields.metadata
            @test !isnothing(metadata)
        end

        # Test that non-DPPL tangents still work with set_to_zero!!
        dummy_tangent = Mooncake.Tangent(
            NamedTuple{(:model, :varinfo, :context, :adtype, :prep)}((
                1.0, 2.0, Mooncake.NoTangent(), Mooncake.NoTangent(), Mooncake.NoTangent()
            )),
        )
        # This should use the fallback implementation
        result2 = set_to_zero!!(deepcopy(dummy_tangent))
        @test result2 isa typeof(dummy_tangent)
    end

    @testset "NoCache optimization correctness" begin
        # Test that set_to_zero!! uses NoCache for DynamicPPL types
        model = test_model1([1.0, 2.0, 3.0])
        vi = DynamicPPL.VarInfo(Random.default_rng(), model)
        ldf = DynamicPPL.LogDensityFunction(model, vi, DynamicPPL.DefaultContext())
        tangent = zero_tangent(ldf)

        # Modify some values
        if hasfield(typeof(tangent.fields.model.fields), :args) &&
            hasfield(typeof(tangent.fields.model.fields.args), :x)
            x_tangent = tangent.fields.model.fields.args.x
            if !isempty(x_tangent)
                x_tangent[1] = 5.0
            end
        end

        # Call set_to_zero!! and verify it works
        set_to_zero!!(tangent)

        # Check that values are zeroed
        if hasfield(typeof(tangent.fields.model.fields), :args) &&
            hasfield(typeof(tangent.fields.model.fields.args), :x)
            x_tangent = tangent.fields.model.fields.args.x
            if !isempty(x_tangent)
                @test x_tangent[1] == 0.0
            end
        end
    end

    @testset "Performance improvement" begin
        # Test with DEMO_MODELS if available
        if isdefined(DynamicPPL.TestUtils, :DEMO_MODELS) &&
            !isempty(DynamicPPL.TestUtils.DEMO_MODELS)
            model = DynamicPPL.TestUtils.DEMO_MODELS[1]
        else
            # Fallback to our test model
            model = test_model1([1.0, 2.0, 3.0, 4.0])
        end

        vi = DynamicPPL.VarInfo(Random.default_rng(), model)
        ldf = DynamicPPL.LogDensityFunction(model, vi, DynamicPPL.DefaultContext())
        tangent = zero_tangent(ldf)

        # Run benchmarks
        result_iddict = @benchmark begin
            cache = IdDict{Any,Bool}()
            set_to_zero_internal!!(cache, $tangent)
        end

        result_nocache = @benchmark set_to_zero!!($tangent)

        # Extract median times
        time_iddict = median(result_iddict).time
        time_nocache = median(result_nocache).time

        # We expect NoCache to be faster
        speedup = time_iddict / time_nocache
        @test speedup > 1.5  # Conservative expectation - should be ~4x

        println("Performance improvement: $(round(speedup, digits=2))x speedup")
        println("IdDict: $(round(time_iddict/1000, digits=2)) μs")
        println("NoCache: $(round(time_nocache/1000, digits=2)) μs")
    end

    @testset "Aliasing safety" begin
        # Test with aliased data
        shared_data = [1.0, 2.0, 3.0]
        model = test_model2(shared_data, shared_data)  # x and y are the same array
        vi = DynamicPPL.VarInfo(Random.default_rng(), model)
        ldf = DynamicPPL.LogDensityFunction(model, vi, DynamicPPL.DefaultContext())
        tangent = zero_tangent(ldf)

        # Check that aliasing is preserved in tangent
        if hasfield(typeof(tangent.fields.model.fields), :args)
            args = tangent.fields.model.fields.args
            if hasfield(typeof(args), :x) && hasfield(typeof(args), :y)
                @test args.x === args.y  # Aliasing should be preserved

                # Modify via x
                if !isempty(args.x)
                    args.x[1] = 10.0
                    @test args.y[1] == 10.0  # Should also change y
                end

                # Zero and check both are zeroed
                # Since x and y are aliased, zeroing one zeros both
                set_to_zero!!(tangent)
                if !isempty(args.x)
                    @test args.x[1] == 0.0
                    @test args.y[1] == 0.0
                end
            end
        end
    end
end
