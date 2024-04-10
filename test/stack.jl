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
end
