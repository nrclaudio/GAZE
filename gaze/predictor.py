"""Inference using the exact implementation and inputs of a versioned release."""
from pathlib import Path
import hashlib,json,tempfile
import numpy as np
import pandas as pd
import torch
from ._reference.config import ExperimentConfig
from ._reference.models.factory import create_model
from ._reference.data.dataset_smiles import FiLMDatasetWithSMILES

class GAZEPredictor:
    """Frozen released model. RNA inputs must already be on the training-compatible scale.

    Use ``from_release`` with the downloaded reproducibility ``reproducibility`` directory.
    The bundled 867-sample deployment weights are separate from the cohort-excluding
    checkpoints used here.
    """
    @classmethod
    def from_release(cls, directory, mode='cohort', device='cpu'):
        if mode not in ('cohort','cohort_plus_targets'):
            raise ValueError('Unknown checkpoint mode')
        self=cls();self.root=Path(directory).resolve();self.device=torch.device(device)
        self.art=self.root/'external/artifacts';self.run=self.root/'external/runs'/mode
        manifest=json.loads((self.root/'release_manifest.json').read_text())['files']
        for p in [self.art/'graph.pt',self.art/'gene_mapping.json',self.art/'camp_reanalysis.h5mu',self.run/'phase2/best_model.pt',self.run/'phase2/experiment_config.json']:
            with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=manifest[str(p.relative_to(self.root))]:raise ValueError('Artifact differs: '+str(p))
        self.config=ExperimentConfig.from_dict(json.loads((self.run/'phase2/experiment_config.json').read_text()))
        self.model=None
        return self

    def _dataset(self,path):
        return FiLMDatasetWithSMILES(str(path),str(self.art/'graph.pt'),str(self.art/'gene_mapping.json'),device=self.device)

    @torch.no_grad()
    def predict_mudata(self,path,rows=None):
        """Predict graph metabolite nodes from compatible MuData; outcomes are masked.

        Returns a DataFrame indexed by input sample ID, columns are graph node IDs.
        Required schema is documented in the release tutorial.
        """
        dataset=self._dataset(path)
        if self.model is None:
            self.model=create_model(self.config.model,embedding_dim=dataset.embedding_dim,gene_dim=dataset.gene_dim,smiles_dim=dataset.smiles_dim).to(self.device)
            self.model.load_state_dict(torch.load(self.run/'phase2/best_model.pt',map_location=self.device,weights_only=True),strict=True)
            self.model.eval()
        rows=list(range(len(dataset))) if rows is None else list(rows)
        if not rows:raise ValueError('No samples selected')
        nodes=dataset.metabolite_indices;pred=[]
        for i in rows:
            sample=dataset[int(i)].to(self.device)
            sample.y[:]=float('nan');sample.mask[:]=False
            values=self.model(sample,compute_physics=False)[0].reshape(-1)[nodes].cpu().numpy()
            if not np.isfinite(values).all():raise ValueError('Nonfinite prediction')
            pred.append(values)
        return pd.DataFrame(pred,index=dataset.mudata['rna'].obs_names[rows],columns=nodes)

    def predict_rna(self,rna):
        """Predict a DataFrame (samples x Ensembl genes) without metabolite measurements.

        Values must use training-compatible preprocessing. This function does not
        estimate normalisation parameters from the supplied samples.
        All genes in the released RNA input are required. NaN denotes a missing
        measurement and leaves its graph feature at zero, as in training.
        """
        import anndata as ad
        import mudata as mu
        template=mu.read_h5mu(self.art/'camp_reanalysis.h5mu')
        required=template['rna'].var['ensembl_gene_id'].astype(str).tolist()
        if not rna.columns.is_unique or not rna.index.is_unique:raise ValueError('RNA identifiers must be unique')
        missing=set(required)-set(rna.columns)
        if missing:raise ValueError(f'Missing {len(missing)} required Ensembl gene columns')
        values=rna.loc[:,required].to_numpy(dtype=np.float32)
        if np.isinf(values).any():raise ValueError('RNA values cannot be infinite')
        if not np.isfinite(values).any(axis=1).all():raise ValueError('Every sample needs measured RNA values')
        obs=pd.DataFrame(index=rna.index.astype(str))
        rv=template['rna'].var.copy();mv=template['metabolomics'].var.copy()
        data=mu.MuData({'rna':ad.AnnData(values,obs=obs.copy(),var=rv),
                       'metabolomics':ad.AnnData(np.full((len(obs),len(mv)),np.nan,dtype=np.float32),obs=obs.copy(),var=mv)})
        with tempfile.TemporaryDirectory(prefix='gaze-rna-') as tmp:
            path=Path(tmp)/'input.h5mu';data.write_h5mu(path)
            return self.predict_mudata(path)
