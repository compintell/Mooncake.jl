@testset "iddict" begin
    @testset "IdDict tangent functionality" begin
        p = IdDict(true => 5.0, false => 4.0)
        z = IdDict(true => 3.0, false => 2.0)
        x = IdDict(true => 1.0, false => 1.0)
        y = IdDict(true => 2.0, false => 1.0)
        test_tangent(sr(123456), p, z, x, y)
    end
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:iddict))
end
