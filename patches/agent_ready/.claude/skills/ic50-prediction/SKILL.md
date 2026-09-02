# IC50 Prediction Skill

## Description

This skill predicts drug sensitivity on cancer cell lines. It estimates whether a drug will be **sensitive** or **resistant** on a given cell line, outputting a probability score (0-1) and binary classification.

The model is trained on large-scale drug response databases (GDSC, PRISM) and uses transfer learning from the transcriptome prediction model.

## Required Input Policy

Never call the model without a real `custom_expression` array containing exactly
10085 finite values in the documented training gene order. The API has no random
or placeholder fallback. `cell_line` is metadata only. If a user has not supplied
a profile, request one or use the bundled public-data-derived A549 smoke-test
profile at `examples/api_context_examples.npz` and disclose that context.

## When to Use This Skill

Use this skill when the user asks about:

1. **Drug efficacy/sensitivity**:
   - "Will this drug work on breast cancer cells?"
   - "Is MCF7 sensitive to ibuprofen?"
   - "Predict the IC50 for this compound"

2. **Drug screening**:
   - "Screen these 100 drugs for sensitivity on A549 cells"
   - "Which drugs are most effective against this cell line?"

3. **Cell line selection**:
   - "Which cell lines are sensitive to this drug?"
   - "Compare drug sensitivity across cell lines"

4. **Companion diagnostics**:
   - "Should this patient's tumor respond to this drug?"
   - "Predict treatment response based on tumor profile"

## How It Works

The model:
1. Takes drug SMILES + cell line RNA-seq profile as input
2. Uses frozen pretrained transcriptome model (RNA encoder, Molformer, cross-attention)
3. Passes fused representation through a trainable MLP head
4. Outputs sensitivity probability (0 = resistant, 1 = sensitive)

**Architecture**: DDPPredictor (Drug-Disease Prediction)
- Base: Transcriptome prediction model (frozen)
- Head: 8-layer MLP with batch normalization and dropout
- Output: Binary classification (sensitive/resistant)

## Usage Instructions

### Basic Usage

```python
import numpy as np
from models.ic50_prediction.ic50_inference import IC50Predictor

context = np.load('examples/api_context_examples.npz', allow_pickle=False)

# Initialize predictor
predictor = IC50Predictor(device='cuda')  # or 'cpu'

# Predict drug sensitivity
result = predictor.predict(
    drug_smiles='CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
    cell_line=str(context['cell_id'][0]),
    custom_expression=context['rna'][0],
)

# View results
print(f"Sensitivity probability: {result['probability']:.3f}")
print(f"Prediction: {result['prediction']}")  # 'sensitive' or 'resistant'
print(f"Confidence: {result['confidence']:.3f}")  # 0-1, how confident
```

### Batch Prediction

```python
import numpy as np
context = np.load('examples/api_context_examples.npz', allow_pickle=False)

# Screen multiple drugs
drug_list = [
    'CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
    'CC(=O)Oc1ccccc1C(=O)O',       # Aspirin
    'CCO',                          # Ethanol
]

results = predictor.predict_batch(
    drug_smiles_list=drug_list,
    cell_line=str(context['cell_id'][0]),
    custom_expression=context['rna'][0],
    batch_size=32
)

for smiles, res in zip(drug_list, results):
    if 'error' in res:
        print(f"{smiles}: Error - {res['error']}")
    else:
        print(f"{smiles}: {res['prediction']} (prob={res['probability']:.3f})")
```

### With Custom Transcriptome Data

```python
import numpy as np

# Load your cell line RNA-seq profile (10085 genes)
custom_profile = np.load('my_cell_rnaseq.npy')  # Shape: (10085,)

result = predictor.predict(
    drug_smiles='CC(C)Cc1ccc(cc1)C(C)C(O)=O',
    cell_line='Custom Cell Line',
    custom_expression=custom_profile
)
```

## Output Format

The `predict()` method returns a dictionary:

```python
{
    'probability': 0.732,        # Sensitivity probability (0-1)
    'logit': 0.943,              # Raw logit value before sigmoid
    'prediction': 'sensitive',   # 'sensitive' or 'resistant' (threshold=0.5)
    'confidence': 0.464,         # abs(prob - 0.5) * 2, range 0-1
    'drug_smiles': 'CCO',        # Input SMILES
    'cell_line': 'MCF7'          # Input cell line
}
```

**Interpretation**:
- `probability > 0.5` → Sensitive
- `probability < 0.5` → Resistant
- `confidence` close to 1 → High confidence
- `confidence` close to 0 → Low confidence (borderline case)

## Examples

### Example 1: Single Drug Screening

**User**: "Will aspirin work on A549 lung cancer cells?"

**Your Response**:
```python
from models.ic50_prediction.ic50_inference import IC50Predictor
import numpy as np

predictor = IC50Predictor()
context = np.load('examples/api_context_examples.npz', allow_pickle=False)

result = predictor.predict(
    drug_smiles='CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
    cell_line=str(context['cell_id'][0]),
    custom_expression=context['rna'][0],
)

if result['prediction'] == 'sensitive':
    print(f"✅ Aspirin is predicted to be SENSITIVE on A549 cells")
    print(f"   Probability: {result['probability']:.1%}")
    print(f"   Confidence: {result['confidence']:.1%}")
else:
    print(f"❌ Aspirin is predicted to be RESISTANT on A549 cells")
    print(f"   Probability: {result['probability']:.1%}")
```

### Example 2: Compare Multiple Drugs

**User**: "Which of these drugs is most effective against lung cancer A549 cells: ibuprofen, aspirin, or paracetamol?"

**Your Response**:
```python
import numpy as np
context = np.load('examples/api_context_examples.npz', allow_pickle=False)
predictor = IC50Predictor()

drugs = {
    'Ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
    'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'Paracetamol': 'CC(=O)Nc1ccc(O)cc1'
}

results = []
for name, smiles in drugs.items():
    res = predictor.predict(
        drug_smiles=smiles,
        cell_line=str(context['cell_id'][0]),
        custom_expression=context['rna'][0],
    )
    results.append((name, res['probability'], res['prediction']))

# Sort by probability (descending)
results.sort(key=lambda x: x[1], reverse=True)

print("Ranked by predicted sensitivity:")
for i, (name, prob, pred) in enumerate(results, 1):
    print(f"{i}. {name}: {prob:.1%} ({pred})")
```

### Example 3: Integrated with Transcriptome Prediction

**User**: "Predict how this drug changes gene expression and whether it will be effective on A549"

**Your Response**:
```python
import numpy as np
context = np.load('examples/api_context_examples.npz', allow_pickle=False)

# First predict transcriptome changes
from models.transcriptome_prediction.transcriptome_inference import TranscriptomePredictor
trans_predictor = TranscriptomePredictor(model_type='l1000', perturbation='drug')

trans_result = trans_predictor.predict(
    drug_smiles='CC(C)Cc1ccc(cc1)C(C)C(O)=O',
    cell_line=str(context['cell_id'][0]),
    custom_expression=context['rna'][0],
    control_expression=context['control'][0],
)

# Then predict sensitivity
from models.ic50_prediction.ic50_inference import IC50Predictor
ic50_predictor = IC50Predictor()

ic50_result = ic50_predictor.predict(
    drug_smiles='CC(C)Cc1ccc(cc1)C(C)C(O)=O',
    cell_line=str(context['cell_id'][0]),
    custom_expression=context['rna'][0],
)

# Combined analysis
print("=== Drug Analysis ===")
print(f"Top affected genes:")
for gene in trans_result['top_genes'][:5]:
    print(f"  - {gene['gene']}: {gene['fold_change']:+.3f}")

print(f"\nPredicted sensitivity: {ic50_result['prediction']}")
print(f"Probability: {ic50_result['probability']:.1%}")
```

## Important Notes

### Current Limitations

1. **Real cell profile is mandatory**:
   - The API never generates a random or placeholder RNA-seq profile
   - Missing, non-finite, or incorrectly shaped input raises `ValueError`
   - Provide `custom_expression`, or use the bundled A549 example only for a smoke test

2. **Cell line names are not validated**:
   - You can pass any string as `cell_line`
   - Model doesn't have built-in cell line database yet
   - Cell line parameter is mainly for metadata tracking

3. **No drug-specific metadata**:
   - Model doesn't know drug names, only SMILES
   - No built-in drugbank ID or chemical name mapping

### Providing Custom Expression Data

To get accurate predictions, provide real cell line RNA-seq data:

```python
import numpy as np

# Your cell line profile should be:
# - 10085 genes (matching the model's RNA encoder)
# - Normalized/log-transformed as appropriate
# - In the same gene order as training data

cell_profile = np.load('MCF7_rnaseq_10085genes.npy')

result = predictor.predict(
    drug_smiles='...',
    cell_line='MCF7',
    custom_expression=cell_profile  # Use real data
)
```

### GPU vs CPU

- **GPU**: Faster, recommended for batch screening
- **CPU**: Works fine for single predictions
- Model is relatively small (~300MB), both are practical

```python
# Use GPU if available
predictor = IC50Predictor(device='cuda')

# Force CPU (e.g., if GPU is busy)
predictor = IC50Predictor(device='cpu')
```

## Error Handling

### Common Errors

1. **CUDA out of memory**:
   ```python
   # Switch to CPU
   predictor = IC50Predictor(device='cpu')
   ```

2. **Invalid SMILES**:
   ```python
   try:
       result = predictor.predict(
           drug_smiles='invalid',
           cell_line='A549',
           custom_expression=cell_profile,
       )
   except Exception as e:
       print(f"Error: {e}")
       print("Please check SMILES validity")
   ```

3. **Custom expression dimension mismatch**:
   ```python
   # Must be exactly 10085 genes
   assert custom_expression.shape == (10085,), "Must provide 10085 gene expression values"
   ```

## Interpretation Guide

### Probability Thresholds

| Probability | Interpretation |
|-------------|----------------|
| 0.9 - 1.0 | Very likely sensitive |
| 0.7 - 0.9 | Likely sensitive |
| 0.5 - 0.7 | Possibly sensitive |
| 0.3 - 0.5 | Possibly resistant |
| 0.1 - 0.3 | Likely resistant |
| 0.0 - 0.1 | Very likely resistant |

### Confidence Scores

| Confidence | Interpretation |
|------------|----------------|
| 0.8 - 1.0 | High confidence - clear prediction |
| 0.5 - 0.8 | Moderate confidence |
| 0.0 - 0.5 | Low confidence - borderline case |

When confidence is low (< 0.5), mention uncertainty:
```python
if result['confidence'] < 0.5:
    print("⚠️  Low confidence prediction - borderline case")
    print("   Consider experimental validation")
```

## Integration with Other Skills

### With Transcriptome Prediction
```python
# Workflow: Predict mechanism → Predict efficacy
trans_result = transcriptome_predictor.predict(
    drug_smiles='...', cell_line='A549', custom_expression=cell_profile,
    control_expression=l1000_control,
)
ic50_result = ic50_predictor.predict(
    drug_smiles='...', cell_line='A549', custom_expression=cell_profile,
)

print(f"Drug affects {len(trans_result['top_genes'])} genes")
print(f"Predicted to be {ic50_result['prediction']}")
```

### With Drug Repurposing
After finding repurposing candidates, validate their predicted efficacy:
```python
import numpy as np
context = np.load('examples/api_context_examples.npz', allow_pickle=False)

# From drug-repurposing skill
top_candidates = moe_predictor.predict_for_disease("Breast cancer", top_n=20)

# Screen with IC50
ic50_predictor = IC50Predictor()
for _, drug in top_candidates.iterrows():
    res = ic50_predictor.predict(
        drug_smiles=drug['smiles'],
        cell_line=str(context['cell_id'][0]),
        custom_expression=context['rna'][0],
    )
    print(f"{drug['drugbank_id']}: {res['prediction']} ({res['probability']:.2f})")
```

## Quick Decision Tree

```
User asks about drug efficacy/sensitivity
    ├─ Do they have drug SMILES?
    │   ├─ YES → Use IC50Predictor ✅
    │   └─ NO → Ask for SMILES or drugbank ID
    │
    ├─ Do they have custom cell line data?
    │   ├─ YES → Use custom_expression parameter
    │   └─ NO → Ask for data; use bundled A549 only for a disclosed smoke test
    │
    └─ Do they want mechanism too?
        ├─ YES → Also use transcriptome-prediction skill
        └─ NO → IC50 prediction alone is sufficient
```

## Files You'll Work With

```
models/ic50_prediction/
├── ic50_inference.py              # Main API - use this
├── ddp_predictor.pt               # Model weights (279 MB)
├── ddp_predict_ic50_new.py        # Model architecture definition
└── dataloader_all.py              # Data loader (for batch training/evaluation)
```

Dependencies (auto-loaded):
```
models/transcriptome_prediction/
├── L1000_vae.pt                   # Pretrained L1000 VAE
├── RNA_vae.pt                     # Pretrained RNA VAE
├── molformer/                     # Pretrained Molformer
└── predictor_L_drug.pt            # Pretrained transcriptome model
```

## Status

- ✅ Model loaded and functional
- ✅ Single and batch prediction working
- ✅ Random/placeholder fallback removed; real expression is mandatory
- ⏳ No built-in cell line database

## Best Practices

1. **Always validate SMILES** before prediction
2. **Mention uncertainty** when confidence is low
3. **Never fabricate a profile**; request real input when `custom_expression` is absent
4. **Combine with transcriptome prediction** for mechanistic insights
5. **Use batch processing** for screening multiple drugs

Last updated: 2026-09-02
