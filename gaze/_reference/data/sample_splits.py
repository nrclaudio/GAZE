"""Validate explicit sample identities before constructing training loaders."""
import hashlib
from pathlib import Path
import pandas as pd


def verify_sha256(path, expected):
    if expected:
        with Path(path).open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'SHA-256 mismatch for {path}; refusing to use an unverified input.')


def manifest_splits(path, rna_obs, met_obs):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'camp_obs_id', 'rna_original_id', 'metabolomics_original_id', 'partition', 'source_group'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Split manifest is missing fields: {sorted(required - set(frame.columns))}')
    for col in ['camp_obs_id', 'rna_original_id', 'metabolomics_original_id']:
        if frame[col].eq('').any() or frame[col].duplicated().any():
            raise ValueError(f'Split manifest has blank or duplicate {col}.')
    if set(frame.partition) - {'train', 'validation', 'holdout', 'excluded'}:
        raise ValueError('Manifest must use finalized train/validation/holdout/excluded partitions.')
    ids = rna_obs.index.astype(str).tolist()
    if ids != met_obs.index.astype(str).tolist() or set(ids) != set(frame.camp_obs_id):
        raise ValueError('Manifest must account for every paired dataset row exactly once.')
    frame = frame.set_index('camp_obs_id').loc[ids]
    for column, obs in [('rna_original_id', rna_obs), ('metabolomics_original_id', met_obs)]:
        if 'original_index' not in obs or frame[column].tolist() != obs.original_index.astype(str).tolist():
            raise ValueError(f'Manifest does not match the actual {column} values.')
    if frame.source_group.eq('').any():
        raise ValueError('Every sample must have an explicit source group.')
    active = frame[frame.partition.isin(['train', 'validation', 'holdout'])]
    if (active.groupby('source_group').partition.nunique() > 1).any():
        raise ValueError('A source group crosses training, validation or holdout partitions.')
    if 'donor_id' in frame:
        known = active[active.donor_id.ne('')]
        if (known.groupby('donor_id').partition.nunique() > 1).any():
            raise ValueError('A known donor crosses training, validation or holdout partitions.')
    train = [i for i, value in enumerate(frame.partition) if value == 'train']
    val = [i for i, value in enumerate(frame.partition) if value == 'validation']
    if not train:
        raise ValueError('Manifest has no training samples.')
    # Holdout is evaluated by a separate command after the model is frozen.
    return {'train': train, 'val': val, 'test': [],
            'holdout': [i for i, value in enumerate(frame.partition) if value == 'holdout'],
            'excluded': [i for i, value in enumerate(frame.partition) if value == 'excluded']}
