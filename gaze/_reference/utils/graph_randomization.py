import torch
import numpy as np

def create_random_graph_version(original_dataset, seed=42):
    """
    Create a randomized version of the graph structure in the dataset
    while preserving node features and labels.
    All samples will share the SAME randomized topology.
    """
    print(f"Creating consistent randomized graph version (seed {seed})...")
    
    if len(original_dataset) == 0:
        return []

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Generate the shared randomized topology using the first sample as a template
    first_sample = original_dataset[0]
    if hasattr(first_sample, 'dataset') and hasattr(first_sample, 'indices'):
        idx = first_sample.indices[0] if isinstance(first_sample.indices, list) else first_sample.indices
        template_data = first_sample.dataset[idx]
    else:
        template_data = first_sample
        
    num_nodes = template_data.num_nodes if hasattr(template_data, 'num_nodes') else template_data.x.size(0)
    num_edges = template_data.edge_index.size(1)
    
    # Generate random source and target nodes (The shared topology)
    random_src = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    random_dst = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    shared_edge_index = torch.stack([random_src, random_dst], dim=0)
    
    # Optionally randomize edge attributes (stoichiometry) if they exist
    shared_edge_attr = None
    if hasattr(template_data, 'edge_attr') and template_data.edge_attr is not None:
        perm = torch.randperm(num_edges)
        shared_edge_attr = template_data.edge_attr[perm]

    # 2. Apply this shared topology to all samples
    random_dataset = []
    for i in range(len(original_dataset)):
        data = original_dataset[i]
        
        # Create a shallow copy instead of deep copy
        if hasattr(data, 'dataset') and hasattr(data, 'indices'):
            idx = data.indices[0] if isinstance(data.indices, list) else data.indices
            original_data = data.dataset[idx]
        else:
            original_data = data
            
        # Create new data object
        new_data = type(original_data)()
        
        # Copy all attributes except graph structure
        for key in original_data.keys():
            if key not in ['edge_index', 'edge_attr']:
                setattr(new_data, key, getattr(original_data, key))
        
        # Inject shared random structure
        new_data.edge_index = shared_edge_index.clone()
        if shared_edge_attr is not None:
            new_data.edge_attr = shared_edge_attr.clone()
            
        random_dataset.append(new_data)
        
    print(f"✅ Randomized {len(random_dataset)} samples with a shared topology ({num_edges} edges).")
    
    return random_dataset

def randomize_graph_edges(data):
    """
    Utility to randomize a single graph object. 
    Maintained for backward compatibility.
    """
    # Create a new data object with the same structure
    new_data = type(data)()
    
    # Copy all attributes except edge_index and edge_attr
    for key in data.keys():
        if key not in ['edge_index', 'edge_attr']:
            setattr(new_data, key, getattr(data, key))
    
    # Randomize the edge indices
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
    num_edges = data.edge_index.size(1)
    
    # Generate random source and target nodes
    random_src = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    random_dst = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    
    # Create new random edge_index
    new_data.edge_index = torch.stack([random_src, random_dst], dim=0)
    
    # Also shuffle edge_attr if they exist to break the stoichiometry-pathway link
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        perm = torch.randperm(num_edges)
        new_data.edge_attr = data.edge_attr[perm]
    
    return new_data 