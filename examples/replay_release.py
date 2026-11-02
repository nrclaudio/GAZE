"""Compare RNA-only package inference with two saved external-cohort predictions."""
from pathlib import Path
import argparse,json
import mudata as mu
import numpy as np
import pandas as pd
import torch
from gaze import GAZEPredictor
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
torch.set_num_threads(2)
root=a.release.resolve();data=mu.read_h5mu(root/'external/artifacts/camp_reanalysis.h5mu')
run=root/'external/runs/cohort/evaluation_primary';meta=pd.read_csv(run/'samples.csv',dtype=str,keep_default_na=False);saved=np.load(run/'predictions.npz')
pos=[np.flatnonzero(meta.holdout_cohort.eq(c))[0] for c in ['RC18','RC20']]
ids=pd.Index(data['rna'].obs.original_index.astype(str));rows=[ids.get_loc(meta.rna_original_id.iloc[i]) for i in pos]
x=data['rna'].X[rows];x=x.toarray() if hasattr(x,'toarray') else np.asarray(x)
rna=pd.DataFrame(x,index=data['rna'].obs_names[rows],columns=data['rna'].var.ensembl_gene_id.astype(str))
predictor=GAZEPredictor.from_release(root);pred=predictor.predict_rna(rna)
actual=pred.loc[:,saved['node_indices']].to_numpy();expected=saved['predictions'][pos]
assert np.allclose(actual,expected,rtol=1e-4,atol=1e-3),'Checkpoint replay mismatch'
record=dict(samples=2,genes=rna.shape[1],targets=actual.shape[1],strict_checkpoint_loading=True,rna_only=True,max_abs_difference=float(np.max(np.abs(actual-expected))),within_tolerance=True)
print(json.dumps(record,indent=2))
if a.output:a.output.write_text(json.dumps(record,indent=2)+'\n')
