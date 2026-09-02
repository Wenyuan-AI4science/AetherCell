# Transcriptome Prediction Skill

## Overview

This skill predicts transcriptome changes (gene expression alterations) caused by drug or gene perturbations, and performs pathway enrichment analysis to identify affected biological pathways.

**Capabilities:**
- 🧬 Predict gene expression changes from **drug treatments** (SMILES input)
- 🔬 Predict effects of **gene perturbations** (knockdown/overexpression/knockout)
- 📊 Perform **pathway enrichment analysis** (GSEA/ORA)
- 🔄 Support both **L1000 platform** (978 genes) and **Bulk RNA-seq** (10085 genes)

## Required Input Policy

Never run a prediction without a real cell profile. The API deliberately raises
an error instead of fabricating one. Every call requires `custom_expression`
(10085 genes); L1000 calls also require the matched `control_expression` (978
genes). `cell_line` is metadata only and does not select a built-in profile. If
the user has not supplied data, ask for it or use the bundled public-data-derived
example at `examples/api_context_examples.npz`, clearly identifying it as A549.

## Quick Start

### When User Asks About Drug Effects

```python
import numpy as np
from models.transcriptome_prediction.transcriptome_inference import TranscriptomePredictor
from models.transcriptome_prediction.pathway_enrichment import PathwayEnricher, print_enrichment_summary

context = np.load('examples/api_context_examples.npz', allow_pickle=False)

# 1. Predict drug effect
predictor = TranscriptomePredictor(
    model_type='bulk_rnaseq',  # full-transcriptome mode; see the L1000 section below
    perturbation='drug',
    device='cuda'
)
result = predictor.predict(
    drug_smiles='CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
    cell_line=str(context['cell_id'][0]),
    custom_expression=context['rna'][0],
)

# 2. Calculate DEG and run enrichment
enricher = PathwayEnricher()
treated = result['expression']
control = treated - result['delta']
enrichment = enricher.enrich_from_bulk_rnaseq(control, treated, method='gsea')

# 3. Show results
print_enrichment_summary(enrichment)
```

### When User Asks About Gene Perturbation Effects

```python
import numpy as np
context = np.load('examples/api_context_examples.npz', allow_pickle=False)

# Predict TP53 knockdown effect
predictor = TranscriptomePredictor(
    model_type='bulk_rnaseq',
    perturbation='shrna',  # knockdown
    device='cuda'
)
result = predictor.predict_gene(gene='TP53', custom_expression=context['rna'][0])

# Run enrichment
enricher = PathwayEnricher()
treated = result['expression']
control = treated - result['delta']
enrichment = enricher.enrich_from_bulk_rnaseq(control, treated, method='gsea')
print_enrichment_summary(enrichment)
```

## Model Selection Guide

| User Need | model_type | perturbation | Output |
|-----------|------------|--------------|--------|
| Drug effect (full genes) | `bulk_rnaseq` | `drug` | 10085 genes |
| Drug effect (L1000) | `l1000` | `drug` | 978 genes |
| Gene knockdown (shRNA) | `bulk_rnaseq` or `l1000` | `shrna` | 10085 or 978 genes |
| Gene overexpression | `bulk_rnaseq` or `l1000` | `oe` | 10085 or 978 genes |
| Gene knockout (CRISPR) | `bulk_rnaseq` or `l1000` | `xpr` | 10085 or 978 genes |

**Recommendation:** Use `bulk_rnaseq` for comprehensive analysis (10085 genes), `l1000` for quick screening (978 genes).

## Input Data Format

### If User Has Custom Expression Data

Tell them to check the example files:
- **Gene order (CRITICAL!):** `models/transcriptome_prediction/examples/gene_list_10085.csv`
- **Expression format:** `models/transcriptome_prediction/examples/example_expression.csv`
- **Drug metadata:** `models/transcriptome_prediction/examples/example_drug_meta.csv`

```python
import pandas as pd
import numpy as np

# Load user's expression data (must match gene order!)
expr_df = pd.read_csv('user_expression.csv', index_col=0)
custom_expression = expr_df.values.flatten()  # (10085,)

# Predict with custom expression
result = predictor.predict(
    drug_smiles='CCO',
    cell_line='Custom Cell Line',
    custom_expression=custom_expression,
)
```

### Common Drug SMILES Examples

| Drug | SMILES |
|------|--------|
| Ibuprofen | `CC(C)Cc1ccc(cc1)C(C)C(O)=O` |
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` |
| Metformin | `CN(C)C(=N)NC(=N)N` |
| Dexamethasone | `CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO` |

### Supported Genes for Perturbation

Common genes are auto-mapped: `TP53`, `BRCA1`, `BRCA2`, `EGFR`, `KRAS`, `MYC`, `PTEN`, `BRAF`, `ERBB2`, `BCL2`

For other genes, use ENSG ID directly (e.g., `ENSG00000141510` for TP53).

## Pathway Enrichment

### DEG Calculation Rules

| Mode | Calculation |
|------|-------------|
| L1000 (978) | `DEG = treated - control` (direct) |
| L1000 expanded (12328) | Expand SEPARATELY, then `DEG = expanded_treated - expanded_control` |
| Bulk RNA-seq (10085) | `DEG = treated - control` (direct) |

### Enrichment Methods

- **GSEA:** Uses all genes ranked by fold change. Best for subtle, coordinated changes.
- **ORA:** Uses DEGs above threshold. Best for strong, specific effects.

### Available Gene Set Databases

- `KEGG_2021_Human` - Metabolic and signaling pathways
- `GO_Biological_Process_2021` - Biological processes
- `GO_Molecular_Function_2021` - Molecular functions
- `Reactome_2022` - Detailed reaction pathways
- `MSigDB_Hallmark_2020` - Cancer hallmark signatures

### L1000 to Full Transcriptome Expansion

If using L1000 mode but need more genes:

```python
# 1. Predict with L1000 mode
import numpy as np
context = np.load('examples/api_context_examples.npz', allow_pickle=False)
predictor = TranscriptomePredictor(model_type='l1000', perturbation='drug')
result = predictor.predict(
    drug_smiles='CCO',
    cell_line=str(context['cell_id'][0]),
    custom_expression=context['rna'][0],
    control_expression=context['control'][0],
)

treated_978 = result['expression']
control_978 = treated_978 - result['delta']

# 2. Expand SEPARATELY (important!)
control_12328 = predictor.expand_l1000_to_full(control_978)['expression']
treated_12328 = predictor.expand_l1000_to_full(treated_978)['expression']

# 3. Run enrichment on expanded
enricher = PathwayEnricher()
enrichment = enricher.enrich_from_l1000_expanded(control_12328, treated_12328)
```

## Complete Example: Drug Mechanism Discovery

**User asks:** "What genes and pathways does Ibuprofen affect?"

```python
import numpy as np
from models.transcriptome_prediction.transcriptome_inference import TranscriptomePredictor
from models.transcriptome_prediction.pathway_enrichment import PathwayEnricher, print_enrichment_summary

context = np.load('examples/api_context_examples.npz', allow_pickle=False)

# Initialize predictor
predictor = TranscriptomePredictor(
    model_type='bulk_rnaseq',
    perturbation='drug',
    device='cuda'
)

# Predict Ibuprofen effect
result = predictor.predict(
    drug_smiles='CC(C)Cc1ccc(cc1)C(C)C(O)=O',
    cell_line=str(context['cell_id'][0]),
    custom_expression=context['rna'][0],
)

print(f"Predicted expression changes for {result['output_dim']} genes")
print("\nTop 10 differentially expressed genes:")
for i, gene in enumerate(result['top_genes'][:10], 1):
    direction = "↑" if gene['fold_change'] > 0 else "↓"
    print(f"  {i}. {direction} {gene['gene']}: {gene['fold_change']:+.3f}")

# Pathway enrichment
enricher = PathwayEnricher()
treated = result['expression']
control = treated - result['delta']

enrichment = enricher.enrich_from_bulk_rnaseq(
    control_expr=control,
    treated_expr=treated,
    method='gsea',
    gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2021']
)

print_enrichment_summary(enrichment)
```

## Files Structure

```
models/transcriptome_prediction/
├── transcriptome_inference.py      # Main API - use this
├── pathway_enrichment.py           # Pathway enrichment (GSEA/ORA)
├── l1000_inference_tool.py         # L1000 → 12328 expansion
├── gene_embedding_loader.py        # Gene embeddings for perturbation
├── examples/                       # ⚠️ Important example files
│   ├── gene_list_10085.csv         # Gene order (CRITICAL!)
│   ├── l1000_gene_order_978.csv    # L1000 gene order
│   ├── example_expression.csv      # Expression data format
│   ├── example_drug_meta.csv       # Drug metadata format
│   ├── example_gene_perturbation_meta.csv
│   ├── pathway_enrichment_example.py  # Complete example
│   └── README.md                   # Detailed documentation
└── [model files and embeddings]
```

## Troubleshooting

### `custom_expression is required` error
No random fallback exists. Supply a real 10085-gene profile in the documented
order. For a smoke test, load `rna` from `examples/api_context_examples.npz`.
For L1000 mode, also supply the matched 978-gene `control` array.

### Gene not found
Use ENSG ID instead of gene symbol, or check if the gene is in the embedding files.

### No significant pathways found
- Try a different gene set database
- Use ORA method with lower FC threshold
- Check if genes are mapped correctly (especially for bulk RNA-seq)

## Status

✅ **All features fully functional:**
- Drug perturbation (L1000 + bulk_rnaseq)
- Gene knockdown (shRNA)
- Gene overexpression (OE)
- Gene knockout (XPR)
- L1000 expansion (978 → 12328)
- Pathway enrichment (GSEA/ORA)
- 10085 ENSG → Symbol mapping

Last updated: 2026-09-02
