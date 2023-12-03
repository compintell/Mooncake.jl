@testset "misc" begin

    @testset "misc utility" begin
        x = randn(4, 5)
        p = Base.unsafe_convert(Ptr{Float64}, x)
        @test Taped.wrap_ptr_as_view(p, 4, 4, 5) == x
        @test Taped.wrap_ptr_as_view(p, 4, 2, 5) == x[1:2, :]
        @test Taped.wrap_ptr_as_view(p, 4, 2, 3) == x[1:2, 1:3]
    end

    TestUtils.run_hand_written_rrule!!_test_cases(StableRNG, Val(:misc))

    @testset "literals" begin
        rng = Xoshiro(123456)
        @testset "Tuple" begin
            x = (5.0, randn(5))
            @test @inferred(lgetfield(x, SInt(1))) == getfield(x, 1)
            test_rrule!!(rng, lgetfield, x, SInt(1); perf_flag=:stability)
            @test @inferred(lgetfield(x, SInt(2))) == getfield(x, 2)
            test_rrule!!(rng, lgetfield, x, SInt(2); perf_flag=:stability)
        end
        @testset "NamedTuple" begin
            x = (a=5.0, b=randn(5))
            @test @inferred(lgetfield(x, SSym(:a))) == getfield(x, :a)
            test_rrule!!(rng, lgetfield, x, SSym(:a); perf_flag=:stability)
            @test @inferred(lgetfield(x, SSym(:b))) == getfield(x, :b)
            test_rrule!!(rng, lgetfield, x, SSym(:b); perf_flag=:stability)
        end
    end
end
