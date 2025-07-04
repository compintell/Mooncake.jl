module MooncakeDynamicPPLExt

using DynamicPPL
using Mooncake
import Mooncake: set_to_zero!!
using Mooncake: NoTangent, Tangent, MutableTangent, NoCache, set_to_zero_internal!!

"""
Check if a tangent corresponds to a DynamicPPL.LogDensityFunction
"""
function is_dppl_ldf_tangent(x)
    x isa Tangent || return false
    hasfield(typeof(x), :fields) || return false

    fields = x.fields
    propertynames(fields) == (:model, :varinfo, :context, :adtype, :prep) || return false

    # Check that model and varinfo are tangents with expected structure
    fields.model isa Tangent || return false
    fields.varinfo isa Tangent || return false

    # Verify model tangent structure
    if hasfield(typeof(fields.model), :fields)
        model_fields = propertynames(fields.model.fields)
        all(f in model_fields for f in (:f, :args, :defaults, :context)) || return false
    end

    # Verify varinfo tangent structure  
    if hasfield(typeof(fields.varinfo), :fields)
        varinfo_fields = propertynames(fields.varinfo.fields)
        all(f in varinfo_fields for f in (:metadata, :logp, :num_produce)) || return false
    end

    return true
end

"""
Check if a tangent corresponds to a DynamicPPL.VarInfo
"""
function is_dppl_varinfo_tangent(x)
    x isa Tangent || return false
    hasfield(typeof(x), :fields) || return false

    fields = x.fields
    propertynames(fields) == (:metadata, :logp, :num_produce) || return false

    # Additional validation could be added here
    return true
end

"""
Check if a tangent corresponds to a DynamicPPL.Model
"""
function is_dppl_model_tangent(x)
    x isa Tangent || return false
    hasfield(typeof(x), :fields) || return false

    fields = x.fields
    all(f in propertynames(fields) for f in (:f, :args, :defaults, :context)) ||
        return false

    return true
end

"""
Check if a MutableTangent corresponds to DynamicPPL.Metadata
"""
function is_dppl_metadata_tangent(x)
    x isa MutableTangent || return false
    hasfield(typeof(x), :fields) || return false

    fields = x.fields
    # Check for the expected fields in Metadata
    expected_fields = (:idcs, :vns, :ranges, :vals, :dists, :orders, :flags)
    all(f in propertynames(fields) for f in expected_fields) || return false

    return true
end

function Mooncake.set_to_zero!!(x)
    # Check if it's a DynamicPPL tangent type that we can optimize
    if x isa Tangent &&
        (is_dppl_ldf_tangent(x) || is_dppl_varinfo_tangent(x) || is_dppl_model_tangent(x))
        return set_to_zero_internal!!(NoCache(), x)
    elseif x isa MutableTangent && is_dppl_metadata_tangent(x)
        return set_to_zero_internal!!(NoCache(), x)
    else
        # Use the original implementation with IdDict for all other types
        return set_to_zero_internal!!(IdDict{Any,Bool}(), x)
    end
end

end # module
