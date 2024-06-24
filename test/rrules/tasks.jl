@testset "tasks" begin
    @testset "Task tangent functionality" begin
        p = Task(() -> nothing)
        x = zero_tangent(p)
        y = zero_tangent(p)
        z = zero_tangent(p)
        TestUtils.test_tangent(sr(123456), p, x, y, z; interface_only=false, perf=false)
        # TestUtils.test_fwds_rvs_data(sr(123456), p)
    end
    TestUtils.run_rrule!!_test_cases(StableRNG, Val(:tasks))
end
