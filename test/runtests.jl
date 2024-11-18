include("front_matter.jl")

@testset "Mooncake.jl" begin
    if test_group == "aqua"
        Aqua.test_all(Mooncake)
    elseif test_group == "basic"
        include("front_matter.jl")
        include("utils.jl")
        include("tangents.jl")
        include("fwds_rvs_data.jl")
        include("codual.jl")
        include("debug_mode.jl")
        include("stack.jl")
        @testset "interpreter" begin
            include(joinpath("interpreter", "contexts.jl"))
            include(joinpath("interpreter", "abstract_interpretation.jl"))
            include(joinpath("interpreter", "bbcode.jl"))
            include(joinpath("interpreter", "ir_utils.jl"))
            include(joinpath("interpreter", "ir_normalisation.jl"))
            include(joinpath("interpreter", "zero_like_rdata.jl"))
            include(joinpath("interpreter", "s2s_reverse_mode_ad.jl"))
        end
        include("tools_for_rules.jl")
        include("interface.jl")
        include("config.jl")
        include("developer_tools.jl")
    elseif test_group == "rrules"
        include("front_matter.jl")
        include("test_utils.jl")
        @testset "rrules" begin
            @info "avoiding_non_differentiable_code"
            include(joinpath("rrules", "avoiding_non_differentiable_code.jl"))
            @info "blas"
            include(joinpath("rrules", "blas.jl"))
            @info "builtins"
            include(joinpath("rrules", "builtins.jl"))
            @info "fastmath"
            include(joinpath("rrules", "fastmath.jl"))
            @info "foreigncall"
            include(joinpath("rrules", "foreigncall.jl"))
            @info "function_wrappers"
            include(joinpath("rrules", "function_wrappers.jl"))
            @info "iddict"
            include(joinpath("rrules", "iddict.jl"))
            @info "lapack"
            include(joinpath("rrules", "lapack.jl"))
            @info "linear_algebra"
            include(joinpath("rrules", "linear_algebra.jl"))
            @info "low_level_maths"
            include(joinpath("rrules", "low_level_maths.jl"))
            @info "misc"
            include(joinpath("rrules", "misc.jl"))
            @info "new"
            include(joinpath("rrules", "new.jl"))
            @info "tasks"
            include(joinpath("rrules", "tasks.jl"))
            @info "twice_precision"
            include(joinpath("rrules", "twice_precision.jl"))
            @static if VERSION >= v"1.11.0-rc4"
                @info "memory"
                include(joinpath("rrules", "memory.jl"))
            end
        end
    else
        throw(error("test_group=$(test_group) is not recognised"))
    end
end
