@testset "stack" begin
    @testset "stack functionality" begin
        s = Stack{Float64}()
        push!(s, 5.0)
        @test s.position == 1
        @test s.memory[1] == 5.0
        @test length(s) == 1
        @test !isempty(s)

        s[] = 6.0
        @test s[] == 6.0
        @test pop!(s) == 6.0
        @test s.position == 0
        @test length(s) == 0
        @test isempty(s)
    end
    @testset "tangent_stack_type" begin
        @test Tapir.tangent_stack_type(Float64) == Stack{Float64}
        @test Tapir.tangent_stack_type(Int) == Tapir.NoTangentStack
        @test Tapir.tangent_stack_type(Any) == Stack{Any}
        @test Tapir.tangent_stack_type(DataType) == Stack{Any}
        @test Tapir.tangent_stack_type(Type{Float64}) == Tapir.NoTangentStack

        @test Tapir.tangent_ref_type_ub(Float64) == Tapir.__array_ref_type(Float64)
        @test Tapir.tangent_ref_type_ub(Int) == Tapir.NoTangentRef
        @test Tapir.tangent_ref_type_ub(Any) == Ref
        @test Tapir.tangent_ref_type_ub(DataType) == Ref
        @test Tapir.tangent_ref_type_ub(Type{Float64}) == Tapir.NoTangentRef
    end
end
