@testset "umlaut_internals_rules" begin

    @testset "misc utility" begin
        x = randn(4, 5)
        p = Base.unsafe_convert(Ptr{Float64}, x)
        @test Taped.wrap_ptr_as_view(p, 4, 4, 5) == x
        @test Taped.wrap_ptr_as_view(p, 4, 2, 5) == x[1:2, :]
        @test Taped.wrap_ptr_as_view(p, 4, 2, 3) == x[1:2, 1:3]
    end

    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:umlaut_internals_rules))
end
