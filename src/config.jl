"""
    Config(; debug_mode::Bool=false, silence_debug_messages::Bool=false)

Configuration struct for use with `ADTypes.AutoMooncake`.

# Keyword Arguments
- `debug_mode::Bool=false`: whether or not to run additional type checks when
    differentiating a function. This has considerable runtime overhead, and should only be
    switched on if you are trying to debug something that has gone wrong in Mooncake.
- `silence_debug_messages::Bool=false`: if `false` and `debug_mode` is `true`, Mooncake will
    display some warnings that debug mode is enabled, in order to help prevent accidentally
    leaving debug mode on. If you wish to disable these messages, set this to `true`.
"""
@kwdef struct Config
    debug_mode::Bool = false
    silence_debug_messages::Bool = false
end
