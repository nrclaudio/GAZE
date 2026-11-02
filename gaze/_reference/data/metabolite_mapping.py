"""Map assay features to graph targets without order-dependent overwrites."""
from collections import defaultdict
import numpy as np
import pandas as pd


def metabolite_feature_nodes(var, hmdb_to_idx, kegg_to_idx):
    nodes = np.full(len(var), -1, dtype=np.int64)
    by_node = defaultdict(list)
    for position, (name, row) in enumerate(var.iterrows()):
        excluded = row.get('exclude_from_prediction', False)
        if pd.notna(excluded) and bool(excluded):
            continue
        hmdb, kegg = row.get('HMDB_trailing_zeros'), row.get('KEGG')
        node = hmdb_to_idx.get(hmdb) if pd.notna(hmdb) else None
        if node is None and pd.notna(kegg):
            node = kegg_to_idx.get(kegg)
        if node is not None:
            nodes[position] = int(node)
            by_node[int(node)].append(str(name))
    collisions = {node: names for node, names in by_node.items() if len(names) > 1}
    if collisions:
        raise ValueError(f'Ambiguous assay features map to the same graph target: {collisions}. '
                         'Resolve the annotations or explicitly exclude the unresolved features before training/evaluation.')
    return nodes


def dense_row(matrix):
    """Support both dense AnnData arrays and sparse matrices."""
    if hasattr(matrix, 'toarray'):
        matrix = matrix.toarray()
    return np.asarray(matrix).reshape(-1)
