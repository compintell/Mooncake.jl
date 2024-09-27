@testset "low_level_maths" begin
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:low_level_maths))

    # We do not want to make this a primitive, because Base.sub_float is sufficient.
    @test !Mooncake.is_primitive(DefaultCtx, Tuple{typeof(-), Float16, Float16})
    @test !Mooncake.is_primitive(DefaultCtx, Tuple{typeof(-), Float32, Float32})
    @test !Mooncake.is_primitive(DefaultCtx, Tuple{typeof(-), Float64, Float64})
end    
