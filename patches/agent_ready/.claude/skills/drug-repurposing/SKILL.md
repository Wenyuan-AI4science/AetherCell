# Drug Repurposing Skill

## Description

This skill predicts the best drug candidates for a given disease using a Mixture-of-Experts (MOE) model. It combines:
- **Transcriptome Expert (TE)**: Measures disease-drug expression similarity
- **Knowledge Graph Expert (KG)**: Leverages disease-drug relationships from biomedical knowledge graphs

The model ranks ~3000+ FDA-approved drugs by their potential effectiveness for ~1000+ diseases.

## When to Use This Skill

Use this skill when the user asks about:

1. **Drug repurposing**:
   - "What drugs could treat Alzheimer's disease?"
   - "Find drug candidates for breast cancer"
   - "Repurpose existing drugs for COVID-19"

2. **Disease-drug matching**:
   - "Which FDA-approved drugs target this disease?"
   - "Rank drugs by effectiveness for lung cancer"

3. **Novel indications**:
   - "What else could metformin be used for?"
   - "Find new uses for aspirin"

4. **Comparing approaches**:
   - "How do transcriptome-based and knowledge graph-based predictions differ?"

## How It Works

**Mixture-of-Experts Architecture**:
1. **Transcriptome Expert (TE)**:
   - Compares disease expression profile with drug-induced expression changes
   - Higher score = drug expression more similar to disease state

2. **Knowledge Graph Expert (KG)**:
   - Uses graph neural network (GNN) embeddings
   - Captures drug-disease relationships from literature, clinical trials, pathways

3. **Gating Network**:
   - Dynamically weights TE and KG experts based on input
   - Outputs final MOE score (fusion of both experts)

**Two Operating Modes**:
- **Full Mode**: Uses both TE and KG (when disease has transcriptome data)
- **KG-Only Mode**: Uses only KG expert (when transcriptome is missing)

## Usage Instructions

### Basic Usage

```python
from models.moe_repurposing.moe_inference import MoEPredictor

# Initialize predictor
predictor = MoEPredictor()

# Predict top drug candidates for a disease
results = predictor.predict_for_disease(
    disease_name="Breast cancer",
    top_n=20  # Return top 20 candidates
)

# View results
print(results[['drugbank_id', 'smiles', 'moe_score', 'te_score', 'kg_score']].head(10))
```

### View Available Diseases

```python
# Get list of all available diseases
diseases = predictor.get_available_diseases()

print(f"Total diseases: {len(diseases)}")
print("\nFirst 10 diseases:")
print(diseases[['disease', 'Mondb_id']].head(10))

# Check if a specific disease is available
if 'Breast cancer' in diseases['disease'].values:
    print("✅ Breast cancer is available")
```

### With Custom Transcriptome Data

```python
# If you have your own disease expression profile
results = predictor.predict_for_disease(
    disease_name="Custom Disease",
    top_n=50,
    custom_expr_file='path/to/disease_expression.csv'
)
```

### Knowledge Graph Only (No Transcriptome)

```python
# For diseases without transcriptome data
results = predictor.predict_kg_only(
    disease_name="Rare Disease",
    top_n=30
)

# Results will be sorted by kg_score instead of moe_score
```

## Output Format

Returns a pandas DataFrame with:

| Column | Description |
|--------|-------------|
| `drugbank_id` | DrugBank identifier (e.g., DB00316) |
| `smiles` | Drug SMILES string |
| `moe_score` | MOE fusion score (higher = better) |
| `te_score` | Transcriptome expert score |
| `kg_score` | Knowledge graph expert score |
| `te_weight` | Weight assigned to TE expert (0-1) |
| `input_status` | "Transcriptome_and_KG" or "KG_Only" |

**Sorting**:
- If transcriptome available: Sorted by `moe_score` (descending)
- If no transcriptome: Sorted by `kg_score` (descending)

## Examples

### Example 1: Standard Repurposing

**User**: "What are the top 10 drug candidates for treating Alzheimer's disease?"

**Your Response**:
```python
from models.moe_repurposing.moe_inference import MoEPredictor

predictor = MoEPredictor()

# Check if disease is available
diseases = predictor.get_available_diseases()
if 'Alzheimer disease' in diseases['disease'].values:
    results = predictor.predict_for_disease(
        disease_name="Alzheimer disease",
        top_n=10
    )

    print("Top 10 drug candidates for Alzheimer's disease:\n")
    for i, row in results.iterrows():
        print(f"{i+1}. {row['drugbank_id']} (SMILES: {row['smiles'][:30]}...)")
        print(f"   MOE Score: {row['moe_score']:.4f}")
        print(f"   TE: {row['te_score']:.4f}, KG: {row['kg_score']:.4f}")
        print(f"   Mode: {row['input_status']}\n")
else:
    # Try fuzzy matching or list similar diseases
    print("Alzheimer disease not found. Available neurodegenerative diseases:")
    neuro = diseases[diseases['disease'].str.contains('disease', case=False)]
    print(neuro['disease'].head(10))
```

### Example 2: Compare MOE Components

**User**: "How do the transcriptome and knowledge graph approaches differ for breast cancer drugs?"

**Your Response**:
```python
predictor = MoEPredictor()

results = predictor.predict_for_disease("Breast cancer", top_n=20)

import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(results['te_score'], results['kg_score'], alpha=0.6)
plt.xlabel('Transcriptome Score (TE)')
plt.ylabel('Knowledge Graph Score (KG)')
plt.title('Expert Score Comparison for Breast Cancer')

# Annotate top drugs
for i in range(5):
    plt.annotate(results.iloc[i]['drugbank_id'],
                 (results.iloc[i]['te_score'], results.iloc[i]['kg_score']))

plt.savefig('expert_comparison.png')

# Correlation
import numpy as np
corr = np.corrcoef(results['te_score'], results['kg_score'])[0, 1]
print(f"TE-KG correlation: {corr:.3f}")

# Identify discrepancies
results['score_diff'] = abs(results['te_score'] - results['kg_score'])
top_diff = results.nlargest(5, 'score_diff')

print("\nDrugs with largest TE-KG disagreement:")
print(top_diff[['drugbank_id', 'te_score', 'kg_score', 'score_diff']])
```

### Example 3: Integrated Workflow

**User**: "Find drugs for lung cancer, then predict their effectiveness on A549 cells"

**Your Response**:
```python
# Step 1: Find repurposing candidates
from models.moe_repurposing.moe_inference import MoEPredictor
moe_predictor = MoEPredictor()

candidates = moe_predictor.predict_for_disease(
    disease_name="Lung cancer",
    top_n=20
)

print("Top 20 repurposing candidates found\n")

# Step 2: Predict IC50 on A549 lung cancer cell line
import numpy as np
from models.ic50_prediction.ic50_inference import IC50Predictor
ic50_predictor = IC50Predictor()
context = np.load('examples/api_context_examples.npz', allow_pickle=False)

sensitivity_results = []
for _, drug in candidates.iterrows():
    try:
        res = ic50_predictor.predict(
            drug_smiles=drug['smiles'],
            cell_line=str(context['cell_id'][0]),
            custom_expression=context['rna'][0],
        )
        sensitivity_results.append({
            'drugbank_id': drug['drugbank_id'],
            'moe_score': drug['moe_score'],
            'sensitivity_prob': res['probability'],
            'prediction': res['prediction']
        })
    except Exception as e:
        print(f"Error for {drug['drugbank_id']}: {e}")

# Convert to DataFrame and filter
import pandas as pd
df = pd.DataFrame(sensitivity_results)
sensitive_drugs = df[df['prediction'] == 'sensitive'].sort_values('moe_score', ascending=False)

print("\n=== Drugs predicted to be SENSITIVE on A549 cells ===")
print(sensitive_drugs.head(10))
```

### Example 4: Custom Disease Expression

**User**: "I have RNA-seq data from a rare disease. Find drug candidates."

**Your Response**:
```python
predictor = MoEPredictor()

# User provides their own disease expression CSV
# CSV should have: gene IDs as index, two columns (disease_sample, control_sample)
results = predictor.predict_for_disease(
    disease_name="Rare Disease XYZ",
    top_n=30,
    custom_expr_file='rare_disease_rnaseq.csv'
)

print(f"Found {len(results)} ranked candidates using custom transcriptome")
print("\nTop 10:")
print(results[['drugbank_id', 'moe_score', 'te_score', 'kg_score']].head(10))
```

## Important Notes

### Disease Name Matching

1. **Exact matching required**:
   - Disease names must match exactly as in the database
   - Case-sensitive in some systems
   - Use `get_available_diseases()` to find exact names

2. **Common diseases available**:
   - "Breast cancer"
   - "Lung cancer"
   - "Alzheimer disease"
   - "Diabetes mellitus"
   - Check the full list with `get_available_diseases()`

3. **If disease not found**:
   ```python
   diseases = predictor.get_available_diseases()

   # Fuzzy search
   query = "alzheimer"
   matches = diseases[diseases['disease'].str.contains(query, case=False)]
   print("Did you mean:")
   print(matches['disease'].tolist())
   ```

### Understanding Scores

**MOE Score** (Fusion):
- Weighted combination of TE and KG scores
- Higher = better predicted efficacy
- Range varies (not normalized 0-1)

**TE Score** (Transcriptome):
- Similarity between disease profile and drug-induced changes
- Only meaningful when transcriptome data is available

**KG Score** (Knowledge Graph):
- Based on disease-drug graph relationships
- Always available (doesn't require transcriptome)

**TE Weight**:
- Confidence assigned to transcriptome expert
- High weight → Model trusts TE more
- Low weight → Model trusts KG more

### Two Operating Modes

**Full Mode** (Transcriptome + KG):
```
input_status: "Transcriptome_and_KG"
Sorting: By moe_score
TE weight: > 0
```

**KG-Only Mode**:
```
input_status: "KG_Only"
Sorting: By kg_score
TE weight: 0
TE score: Not used (set to 0)
```

### Data Coverage

- **Diseases**: ~1000+ with metadata
  - Subset have transcriptome data (full mode)
  - All have KG embeddings (KG mode)

- **Drugs**: ~3000+ FDA-approved
  - All have SMILES
  - All have KG embeddings
  - Subset have GNN scores

## Error Handling

### Disease Not Found

```python
try:
    results = predictor.predict_for_disease("Unknown Disease", top_n=10)
except ValueError as e:
    print(f"Error: {e}")

    # Show available diseases
    diseases = predictor.get_available_diseases()
    print("\nAvailable diseases:")
    print(diseases['disease'].head(20))
```

### Empty Results

```python
results = predictor.predict_for_disease("Disease Name", top_n=50)

if results.empty:
    print("No results returned. Possible issues:")
    print("- Disease has no valid GNN row")
    print("- Expression data missing for this disease")
    print("- Check disease_all_meta.csv for this disease's metadata")
```

### Memory Issues

```python
# Model runs on CPU (no GPU needed)
# If memory issues occur, reduce batch size in the source code
# or process fewer drugs at once

# MOE inference is CPU-bound, ~1GB RAM should be sufficient
```

## Integration with Other Skills

### With IC50 Prediction
Filter repurposing candidates by predicted cell line sensitivity:

```python
# 1. Find candidates
candidates = moe_predictor.predict_for_disease("Disease", top_n=50)

# 2. Screen with IC50
import numpy as np
from models.ic50_prediction.ic50_inference import IC50Predictor
ic50 = IC50Predictor()
context = np.load('examples/api_context_examples.npz', allow_pickle=False)

for _, drug in candidates.iterrows():
    res = ic50.predict(
        drug['smiles'],
        cell_line=str(context['cell_id'][0]),
        custom_expression=context['rna'][0],
    )
    if res['prediction'] == 'sensitive':
        print(f"✅ {drug['drugbank_id']}: MOE={drug['moe_score']:.3f}, IC50={res['probability']:.3f}")
```

### With Transcriptome Prediction
Validate mechanism of top candidates:

```python
# 1. Find candidates
candidates = moe_predictor.predict_for_disease("Breast cancer", top_n=10)

# 2. Check mechanism in the bundled A549 public-data smoke-test context
import numpy as np
from models.transcriptome_prediction.transcriptome_inference import TranscriptomePredictor
trans = TranscriptomePredictor(model_type='l1000', perturbation='drug')
context = np.load('examples/api_context_examples.npz', allow_pickle=False)

for _, drug in candidates.head(3).iterrows():
    print(f"\n=== {drug['drugbank_id']} ===")
    result = trans.predict(
        drug['smiles'],
        cell_line=str(context['cell_id'][0]),
        custom_expression=context['rna'][0],
        control_expression=context['control'][0],
    )
    print("Top affected genes:")
    for gene in result['top_genes'][:5]:
        print(f"  {gene['gene']}: {gene['fold_change']:+.3f}")
```

## Quick Decision Tree

```
User asks about drug repurposing
    ├─ Do they specify a disease?
    │   ├─ YES → Check if disease in database
    │   │   ├─ Found → Run prediction ✅
    │   │   └─ Not found → Show available diseases, ask for clarification
    │   └─ NO → Ask which disease they want to target
    │
    ├─ Do they have custom transcriptome data?
    │   ├─ YES → Use custom_expr_file parameter
    │   └─ NO → Use database (full or KG-only mode)
    │
    └─ Do they want detailed analysis?
        ├─ YES → Also use IC50 or transcriptome prediction skills
        └─ NO → Return ranked list only
```

## Files You'll Work With

```
models/moe_repurposing/
├── moe_inference.py                   # Main API - use this
├── standalone_expert_model.pt         # MOE TorchScript model (394 MB)
├── data_sub/                          # Static data (991 MB)
│   ├── disease_all_meta.csv          # ~1000 diseases
│   ├── disease_control_rnaseq_exp_clean.csv  # Expression profiles
│   ├── static_data.h5                # KG embeddings, GNN scores
│   └── drugs_name.csv                # Drug names
├── inference_dataset.py               # Dataset loaders
└── infrence1.py                       # Reference implementation
```

## Performance

- **Inference time**: ~30-60 seconds for full prediction (1 disease × 3000 drugs)
- **Device**: CPU only (TorchScript traced model)
- **Memory**: ~2GB RAM
- **Batch size**: 64 drugs per batch (configurable)

## Best Practices

1. **Always check disease availability first** with `get_available_diseases()`
2. **Mention the mode** (Full vs KG-only) in your response
3. **Interpret scores in context**: MOE score is relative, not absolute
4. **Combine with IC50** for cell-line specific validation
5. **Validate top candidates** with transcriptome prediction for mechanism
6. **Handle errors gracefully** - not all diseases have full data

## Status

- ✅ Model loaded and fully functional
- ✅ ~1000 diseases, ~3000 drugs available
- ✅ Both full and KG-only modes working
- ✅ Custom transcriptome input supported

Last updated: 2025-03-10
