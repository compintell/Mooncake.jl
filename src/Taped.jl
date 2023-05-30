module Taped

using Distributions, Umlaut
import Umlaut: isprimitive, Frame, Tracer

# Core functionality.
include("tracing.jl")

# Functions of tapes.
include("is_pure.jl")

# Specific transformation examples.
include("vmap.jl")
include("forwards_mode_ad.jl")
include("reverse_mode_ad.jl")
include("logpdf.jl")

end
