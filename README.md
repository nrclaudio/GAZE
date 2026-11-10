# GAZE

**Graph Attention for Zero-shot metabolite Estimation**

GAZE is a graph neural network for predicting relative metabolite abundances from gene expression. Its shared Metabolite-Conditioned Reader (MCR) combines chemical descriptors with metabolic-network representations; the main model reads reactions up to two hops from each metabolite (the multi-hop readout). In LOMO zero-shot evaluation, a target's abundance labels are withheld while its graph node and descriptors remain available.

## Installation and tested inference

Use Python 3.12 and the pinned environment from the [reproducibility release](https://github.com/nrclaudio/gaze-reproducibility/releases/tag/v2026.11.11), then install this package:

```bash
pip install .
```

Download the reproducibility release artifacts as explained in its [tutorial](https://github.com/nrclaudio/gaze-reproducibility/blob/main/reproducibility/TUTORIAL.md). Run the example from this repository:

```bash
python examples/replay_release.py --release ../gaze-reproducibility/reproducibility
```

This example explicitly selects `mode="cohort"` to reproduce the external-validation predictions of the main model, GAZE with the multi-hop readout. That selection is not a comparison showing that this model is the best deployment model. The example strictly loads the released 615-sample CAMP model, predicts RNA-only inputs for two excluded-cohort samples, and compares them with saved predictions. The graph, gene mapping, RNA schema, training configuration and weights are checked against the release manifest. No labels or parameter updates are used during inference.

```python
from gaze import GAZEPredictor

predictor = GAZEPredictor.from_release(
    "../gaze-reproducibility/reproducibility", mode="cohort"
)
predictions = predictor.predict_rna(rna_dataframe)
```

`rna_dataframe` has samples as rows and Ensembl gene IDs as columns. Provide every gene column in the released RNA schema, using compatible preprocessing. NaN marks an unmeasured gene and leaves its graph feature at zero, as in training. Infinite values and wholly unmeasured samples are rejected. Raw counts are not automatically transformed into the training scale. Missing gene columns raise an error. Output columns are graph node IDs; names and identifiers are available in the released graph.

`GAZEPredictor` uses the original training implementation under `gaze/_reference`, with import paths namespaced for packaging. Its provenance manifest records the source hashes. The earlier `GAZEModel`/`GAZEConfig` convenience API is retained for compatibility; the validated release workflow uses `GAZEPredictor`.

## Models and graph

The released graph has 3,687 nodes and 6,760 directed edges. The main model combines enzyme expression and functional embeddings, ChemBERTa metabolite descriptors, relational graph attention, cross-modal attention and the shared MCR readout over two-hop reaction neighborhoods. GAZE with biochemical regularization additionally produces auxiliary reaction outputs constrained by stoichiometry and gene expression; these outputs are not independently validated biological fluxes, and the regularization did not improve prediction of held-out metabolites once the multi-hop readout was used.

From reproducibility release v2026.11.11, `from_release` loads the two external-validation checkpoints of the main model, trained after excluding RC18/RC20 and 32 additional renal samples, using 615 development samples. `mode="cohort_plus_targets"` selects the model that additionally excludes 20 metabolite label sets. The corresponding checkpoints of GAZE with biochemical regularization are kept in that release under `external/regularized/` and load with `from_release` from release v2026.11.02.1.

The files in `weights/` contain the original model trained on all 867 CAMP samples, including tissue samples. They come from the earlier implementation, before corrections to attention-cache handling and the prediction path. `GAZEPredictor` currently supports the two corrected external-validation models; the original all-sample checkpoint has not been validated with this loader. A file named `best_model.pt` is the selected checkpoint within its own run, not evidence that it outperforms checkpoints from other runs. An all-sample deployment model using the corrected training procedure remains separate work.

Training, LOMO, ablations, external evaluation, processed data, source attributions and expected numerical results are available in the [reproducibility repository](https://github.com/nrclaudio/gaze-reproducibility).

## Changes

- 0.2.2: the configuration reader accepts the training-only `validation_objective` setting recorded with the main-model checkpoints, so they load directly. The numerical implementation is unchanged.
