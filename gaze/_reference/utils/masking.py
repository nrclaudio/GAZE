
import torch

def get_teleporter_blacklist():
    """
    Returns a set of metabolite names that should be masked during message passing
    to prevent topological short-circuits (hubs/teleporters).
    """
    return {
        # Nucleotides / Energy
        "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "CTP", "CDP", "CMP", "UTP", "UDP", "UMP",
        "PPi", "Pi", "phosphate", "pyrophosphate",
        
        # Redox / Carriers
        "NAD+", "NADH", "NADP+", "NADPH", "FAD", "FADH2", "H2O2", "O2",
        
        # Solvent / Small Molecules
        "H+", "H2O", "CO2", "NH3", "sulfate",
        
        # Empty Carriers
        "CoA"
    }

def get_teleporter_blacklist_ids():
    """
    Returns a set of HMDB IDs for common hub metabolites.
    """
    return {
        # ATP/ADP/AMP
        "HMDB0000538", "HMDB0001341", "HMDB0000045",
        # GTP/GDP/GMP
        "HMDB0001273", "HMDB0001201", "HMDB0001397",
        # CTP/CDP/CMP (CMP: HMDB0000095 is 5'-CMP)
        "HMDB0000082", "HMDB0000300", "HMDB0000095", "HMDB0240312",
        # UTP/UDP/UMP
        "HMDB0000285", "HMDB0000295", "HMDB0000288",
        # Phosphate / Pyrophosphate
        "HMDB0001429", "HMDB0000250",
        # NAD+/NADH
        "HMDB0000902", "HMDB0001487",
        # NADP+/NADPH
        "HMDB0000217", "HMDB0304435", "HMDB0000221",
        # FAD/FADH2
        "HMDB0001248", # FAD
        # H2O / CO2 / NH3
        "HMDB0002111", "HMDB0001967", "HMDB0000051", "HMDB0041827",
        # CoA
        "HMDB0001423",
        # Sulfate
        "HMDB0001448"
    }

def get_protected_whitelist():
    """
    Metabolites that should NEVER be masked even if they have high degree,
    as they carry the primary carbon signal.
    """
    return {
        "acetyl-CoA", "L-carnitine", "malonyl-CoA",
        "glutamate", "AKG", "pyruvate", "lactate",
        "citrate", "malate", "succinate", "fumarate",
        "SAM", "SAH"
    }

def get_protected_whitelist_ids():
    """
    HMDB IDs for protected metabolites.
    """
    return {
        "HMDB0001206", # Acetyl-CoA
        "HMDB0000062", # L-carnitine
        "HMDB0001175", # Malonyl-CoA
        "HMDB0000148", # Glutamate
        "HMDB0000208", # AKG
        "HMDB0000243", # Pyruvate
        "HMDB0000190", # Lactate
        "HMDB0000094", # Citrate
        "HMDB0000156", # Malate
        "HMDB0000254", # Succinate
        "HMDB0000134", # Fumarate
        "HMDB0001185", # SAM
        "HMDB0000939"  # SAH
    }

def create_pathway_edge_mask(edge_index, node_names, node_types, node_hmdb_ids=None):
    """
    Creates a boolean mask for edges that should be used for pathway-sensitive
    message passing.
    
    Rule: Mask an edge if either the source or target is a blacklisted metabolite.
    Exceptions: If the metabolite is in the protected whitelist, we keep it.
    """
    blacklist_names = get_teleporter_blacklist()
    whitelist_names = get_protected_whitelist()
    
    blacklist_ids = get_teleporter_blacklist_ids()
    whitelist_ids = get_protected_whitelist_ids()
    
    num_edges = edge_index.size(1)
    mask = torch.ones(num_edges, dtype=torch.bool)
    
    # Pre-calculate which node indices are blacklisted
    is_blacklisted = torch.zeros(len(node_names), dtype=torch.bool)
    
    for i, (name, ntype) in enumerate(zip(node_names, node_types)):
        if ntype == 'metabolite':
            should_mask = False
            is_protected = False
            
            # 1. Check Names
            upper_name = name.upper()
            
            # Check name blacklist
            for black_item in blacklist_names:
                if black_item.upper() == upper_name:
                    should_mask = True
                    break
            
            # Check name whitelist (overrides blacklist)
            for white_item in whitelist_names:
                if white_item.upper() == upper_name:
                    is_protected = True
                    break
            
            # 2. Check IDs (if available)
            if node_hmdb_ids is not None:
                hmdb_id = node_hmdb_ids[i]
                if hmdb_id:
                    # Check ID blacklist
                    if hmdb_id in blacklist_ids:
                        should_mask = True
                    
                    # Check ID whitelist (overrides everything)
                    if hmdb_id in whitelist_ids:
                        is_protected = True
            
            # Final decision
            if should_mask and not is_protected:
                is_blacklisted[i] = True
    
    # Mask edges where either end is a blacklisted metabolite
    src, dst = edge_index[0], edge_index[1]
    mask = ~(is_blacklisted[src] | is_blacklisted[dst])
    
    return mask
