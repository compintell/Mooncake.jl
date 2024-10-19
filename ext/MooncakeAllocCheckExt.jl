module MooncakeAllocCheckExt

using AllocCheck, Mooncake

import Mooncake.TestUtils: check_allocs, Shim

@check_allocs function check_allocs(::Shim, f::F, x::Tuple{Vararg{Any, N}}) where {F, N}
    return f(x...)
end

end
