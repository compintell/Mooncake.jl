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
        @test Taped.tangent_stack_type(Float64) == Stack{Float64}
        @test Taped.tangent_stack_type(Int) == Taped.NoTangentStack
        @test Taped.tangent_stack_type(Any) == Stack{Any}
        @test Taped.tangent_stack_type(DataType) == Taped.NoTangentStack
        @test Taped.tangent_stack_type(Type{Float64}) == Taped.NoTangentStack

        @test Taped.tangent_ref_type_ub(Float64) == Taped.__array_ref_type(Float64)
        @test Taped.tangent_ref_type_ub(Int) == Taped.NoTangentRef
        @test Taped.tangent_ref_type_ub(Any) == Ref
        @test Taped.tangent_ref_type_ub(DataType) == Ref
        @test Taped.tangent_ref_type_ub(Type{Float64}) == Taped.NoTangentRef
    end
end
