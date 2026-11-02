# GAZE

**Graph Attention for Zero-shot metabolite Estimation**

GAZE is a biochemically regularized graph neural network for predicting relative metabolite abundances from gene expression. Its shared Metabolite-Conditioned Reader combines chemical descriptors with metabolic-network representations. In LOMO zero-shot evaluation, a target's abundance labels are withheld while its graph node and descriptors remain available.

## Installation and tested inference

Use Python 3.12 and the pinned environment from the [reproducibility release](https://github.com/nrclaudio/gaze-reproducibility/releases/tag/v2026.11.02.1), then install this package:

```bash
pip install .
```

Download the reproducibility release artifacts as explained in its [tutorial](https://github.com/nrclaudio/gaze-reproducibility/blob/v2026.11.02.1/reproducibility/TUTORIAL.md). Run the example from this repository:

```bash
python examples/replay_release.py --release ../gaze-reproducibility/reproducibility
```

The example strictly loads the released 615-sample CAMP model, predicts RNA-only inputs for two excluded-cohort samples, and compares them with saved predictions. The graph, gene mapping, RNA schema, training configuration and weights are checked against the release manifest. No labels or parameter updates are used during inference.

```python
from gaze import GAZEPredictor

predictor = GAZEPredictor.from_release("../gaze-reproducibility/reproducibility")
predictions = predictor.predict_rna(rna_dataframe)
```

`rna_dataframe` has samples as rows and Ensembl gene IDs as columns. Provide every gene column in the released RNA schema, using compatible preprocessing. NaN marks an unmeasured gene and leaves its graph feature at zero, as in training. Infinite values and wholly unmeasured samples are rejected. Raw counts are not automatically transformed into the training scale. Missing gene columns raise an error. Output columns are graph node IDs; names and identifiers are available in the released graph.

`GAZEPredictor` uses the original training implementation under `gaze/_reference`, with import paths namespaced for packaging. Its provenance manifest records the source hashes. The earlier `GAZEModel`/`GAZEConfig` convenience API is retained for compatibility; the validated release workflow uses `GAZEPredictor`.

## Models and graph

The released graph has 3,687 nodes and 6,760 directed edges. It combines enzyme expression and functional embeddings, ChemBERTa metabolite descriptors, relational graph attention, cross-modal attention and the shared MCR readout. Auxiliary reaction outputs support biochemical regularization; they are not independently validated biological fluxes.

The versioned reproducibility download includes two external-validation checkpoints trained after excluding RC18/RC20 and 32 additional renal samples, using 615 development samples. `mode="cohort_plus_targets"` selects the model that additionally excludes 20 metabolite label sets.

The older files in `weights/` were trained on 867 CAMP samples, including tissue samples. They are a separate deployment model and must not be substituted for the cohort-excluding checkpoints. They are preserved unchanged for provenance.

Training, LOMO, ablations, external evaluation, processed data, source attributions and expected numerical results are available in the [reproducibility repository](https://github.com/nrclaudio/gaze-reproducibility).
