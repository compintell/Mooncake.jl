module ContextsTestModule

using Mooncake: @is_primitive, DefaultCtx

foo(x) = x

@is_primitive DefaultCtx Tuple{typeof(foo),Float64}

end

@testset "contexts" begin
    @test Mooncake.is_primitive(DefaultCtx, Tuple{typeof(ContextsTestModule.foo),Float64})
    @test !Mooncake.is_primitive(DefaultCtx, Tuple{typeof(ContextsTestModule.foo),Real})
end
