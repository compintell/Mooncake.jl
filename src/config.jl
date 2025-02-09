"""
    Config(; debug_mode=false, silence_debug_messages=false)

Configuration struct for use with ADTypes.AutoMooncake.
"""
@kwdef struct Config
    debug_mode::Bool = false
    silence_debug_messages::Bool = false
end
