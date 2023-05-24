module Taped

using Umlaut
import Umlaut: isprimitive, Frame, Tracer

# Core functionality.
include("tracing.jl")

# Functions of tapes.
include("is_pure.jl")

# Specific transformation examples.
include("vmap.jl")
include("forwards_mode_ad.jl")

# include("algorithmic_differentiation.jl")

end
