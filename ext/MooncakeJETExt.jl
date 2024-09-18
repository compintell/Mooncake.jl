module MooncakeJETExt

    using JET, Mooncake

    Mooncake.TestUtils.test_opt(::Mooncake.TestUtils.Shim, args...) = JET.test_opt(args...)
    Mooncake.TestUtils.report_opt(::Mooncake.TestUtils.Shim, tt) = JET.report_opt(tt)
end
