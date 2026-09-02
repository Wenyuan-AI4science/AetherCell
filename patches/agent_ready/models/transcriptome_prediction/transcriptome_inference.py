"""
Simplified Transcriptome Prediction Inference API

This module provides a clean, easy-to-use interface for predicting transcriptome
changes from drug/gene perturbations without requiring deep knowledge of the
underlying model architecture.

Supports two output modes:
- L1000: 978-dimensional output (landmark genes)
- Bulk RNA-seq: 10085-dimensional output (full transcriptome)

Usage:
    from transcriptome_inference import TranscriptomePredictor
    import numpy as np

    context = np.load('examples/api_context_examples.npz')
    rna_profile = context['rna'][0]       # 10085-gene real cell profile
    l1000_control = context['control'][0] # 978-gene matched control

    # L1000 mode (978 genes output)
    predictor = TranscriptomePredictor(model_type='l1000', perturbation='drug')
    result = predictor.predict(
        drug_smiles='CCO',
        cell_line=str(context['cell_id'][0]),
        custom_expression=rna_profile,
        control_expression=l1000_control,
    )

    # Bulk RNA-seq mode (10085 genes output)
    predictor = TranscriptomePredictor(model_type='bulk_rnaseq', perturbation='drug')

The API never synthesizes a cell profile. A real 10085-gene RNA profile is
required for every prediction, and L1000 mode additionally requires its matched
978-gene control profile.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from transformers import AutoModel, AutoTokenizer


class TranscriptomePredictor:
    """Unified interface for transcriptome prediction models"""

    def __init__(self,
                 model_type: str = 'l1000',
                 perturbation: str = 'drug',
                 model_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the predictor

        Args:
            model_type: 'l1000' (978 genes) or 'bulk_rnaseq' (10085 genes)
            perturbation: 'drug', 'shrna', 'oe' (overexpression), or 'xpr' (knockout)
            model_dir: Directory containing model files (auto-detected if None)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_type = model_type.lower()
        self.perturbation = perturbation.lower()

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Find model directory
        if model_dir is None:
            self.model_dir = Path(__file__).parent
        else:
            self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # Determine output dimension
        self.output_dim = 10085 if self.model_type == 'bulk_rnaseq' else 978

        # Load models
        print(f"Loading {model_type} + {perturbation} models...")
        print(f"  Output dimension: {self.output_dim}")
        self._load_models()
        self._enforce_deterministic_eval(self.predictor)
        print(f"Models loaded on {self.device}")

    @staticmethod
    def _enforce_deterministic_eval(module: nn.Module) -> None:
        """Freeze MolFormer's random-feature maps after checkpoint loading."""
        module.eval()
        for child in module.modules():
            if hasattr(child, "deterministic") and hasattr(child, "orthogonal_random_weights"):
                child.deterministic = True

    @staticmethod
    def _validated_profile(value: Optional[np.ndarray], size: int, name: str) -> np.ndarray:
        """Validate a real, one-dimensional expression profile without inventing data."""
        if value is None:
            example = "examples/api_context_examples.npz"
            raise ValueError(
                f"{name} is required; AetherCell never substitutes a random cell profile. "
                f"Provide a real {size}-gene array in the documented gene order. "
                f"A public-data-derived example is available at {example}."
            )
        profile = np.asarray(value, dtype=np.float32)
        if profile.ndim != 1 or profile.shape[0] != size:
            raise ValueError(f"{name} must be a one-dimensional array with {size} genes, got {profile.shape}")
        if not np.isfinite(profile).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return np.ascontiguousarray(profile)

    def _load_models(self):
        """Load all required model components"""
        import sys
        sys.path.insert(0, str(self.model_dir))

        from LINCSvae import LINCSVAE
        from RNAvae import RNAVAE

        # Load VAE models
        print("  Loading VAE models...")

        # L1000 VAE
        l1000_vae_path = self.model_dir / "L1000_vae.pt"
        LINvae_model = LINCSVAE("cpu")
        lin_ckpt = torch.load(l1000_vae_path, map_location="cpu", weights_only=False)
        LINvae_model.load_state_dict(lin_ckpt['vae_model_state_dict'])
        self.l1000_encoder = LINvae_model.encoder.to(self.device).eval()
        self.l1000_decoder = LINvae_model.decoder.to(self.device).eval()

        # RNA VAE (needed for both encoding and bulk_rnaseq decoding)
        rna_vae_path = self.model_dir / "RNA_vae.pt"
        self.RNAvae_model = RNAVAE("cpu")
        rna_ckpt = torch.load(rna_vae_path, map_location="cpu", weights_only=False)
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in rna_ckpt['vae_model_state_dict'].items()}
        self.RNAvae_model.load_state_dict(state_dict)
        self.rna_encoder = self.RNAvae_model.encoder.to(self.device).eval()
        self.rna_decoder = self.RNAvae_model.decoder.to(self.device).eval()

        # Load perturbation predictor
        print(f"  Loading {self.model_type} + {self.perturbation} predictor...")

        # Map model type and perturbation to checkpoint file
        # Note: All checkpoints use the same base architecture (L1000 decoder)
        # For bulk_rnaseq mode, we wrap them with R_decoder
        predictor_map = {
            ('l1000', 'drug'): 'predictor_L_drug.pt',
            ('l1000', 'shrna'): 'predictor_L_sh.pt',
            ('l1000', 'sh'): 'predictor_L_sh.pt',
            ('l1000', 'oe'): 'predictor_L_oe.pt',
            ('l1000', 'overexpression'): 'predictor_L_oe.pt',
            ('l1000', 'xpr'): 'predictor_L_xpr.pt',
            ('l1000', 'knockout'): 'predictor_L_xpr.pt',
            ('bulk_rnaseq', 'drug'): 'predictor_R_drug.pt',
            ('bulk_rnaseq', 'shrna'): 'predictor_R_sh.pt',
            ('bulk_rnaseq', 'sh'): 'predictor_R_sh.pt',
            ('bulk_rnaseq', 'oe'): 'predictor_R_oe.pt',
            ('bulk_rnaseq', 'overexpression'): 'predictor_R_oe.pt',
            ('bulk_rnaseq', 'xpr'): 'predictor_R_xpr.pt',
            ('bulk_rnaseq', 'knockout'): 'predictor_R_xpr.pt',
        }

        predictor_file = predictor_map.get((self.model_type, self.perturbation))
        if not predictor_file:
            raise ValueError(f"Unsupported combination: {self.model_type} + {self.perturbation}")

        predictor_path = self.model_dir / predictor_file

        if not predictor_path.exists():
            raise FileNotFoundError(f"Predictor model not found: {predictor_path}")

        # Load Molformer tokenizer
        molformer_path = self.model_dir / "molformer"
        self.tokenizer = AutoTokenizer.from_pretrained(molformer_path, trust_remote_code=True)

        # Load the specific predictor based on type
        if self.perturbation == 'drug':
            self._load_drug_predictor(predictor_path, molformer_path)
        else:
            self._load_gene_predictor(predictor_path)

        print("  All models loaded")

    def _load_drug_predictor(self, predictor_path: Path, molformer_path: Path):
        """Load drug perturbation predictor"""
        from uniperturb_drug import JointPerturbationPredictor

        # Build base predictor
        base_model = JointPerturbationPredictor(
            self.l1000_encoder,
            self.l1000_decoder,
            self.rna_encoder,
            str(molformer_path),
            self.device
        ).to(self.device)

        # Load checkpoint
        predict_ckpt = torch.load(predictor_path, map_location=self.device, weights_only=False)
        state_dict = {k.replace('module.', ''): v for k, v in predict_ckpt['model_state_dict'].items()}
        base_model.load_state_dict(state_dict)

        if self.model_type == 'bulk_rnaseq':
            # Wrap with R_decoder for 10085 output
            from uniperturb_R_drug import RNADrugResponsePredictor
            self.predictor = RNADrugResponsePredictor(
                pretrained_model=base_model,
                RNAVAE=self.RNAvae_model,
                device=self.device
            ).to(self.device)
        else:
            # Use base model with L_decoder for 978 output
            self.predictor = base_model

        self.predictor.eval()

    def _load_gene_predictor(self, predictor_path: Path):
        """Load gene perturbation predictor (shRNA/OE/XPR)"""
        from uniperturb_sh import JointPerturbationPredictor_sh

        # Build base predictor
        base_model = JointPerturbationPredictor_sh(
            self.l1000_encoder,
            self.l1000_decoder,
            self.rna_encoder,
            self.device
        ).to(self.device)

        # Load checkpoint
        predict_ckpt = torch.load(predictor_path, map_location=self.device, weights_only=False)
        state_dict = {k.replace('module.', ''): v for k, v in predict_ckpt['model_state_dict'].items()}
        base_model.load_state_dict(state_dict)

        if self.model_type == 'bulk_rnaseq':
            # Wrap with R_decoder for 10085 output
            from uniperturb_R_sh import RNAGeneResponsePredictor
            self.predictor = RNAGeneResponsePredictor(
                pretrained_model=base_model,
                RNAVAE=self.RNAvae_model,
                device=self.device
            ).to(self.device)
        else:
            # Use base model with L_decoder for 978 output
            self.predictor = base_model

        self.predictor.eval()

    def predict(self,
                drug_smiles: Optional[str] = None,
                gene_embedding_ppi: Optional[np.ndarray] = None,
                gene_embedding_seq: Optional[np.ndarray] = None,
                cell_line: str = 'MCF7',
                custom_expression: Optional[np.ndarray] = None,
                control_expression: Optional[np.ndarray] = None,
                return_latent: bool = False) -> Dict:
        """
        Predict transcriptome changes

        Args:
            drug_smiles: SMILES string for drug perturbation
            gene_embedding_ppi: Gene PPI embedding (256-dim) for gene perturbation
            gene_embedding_seq: Gene sequence embedding (1152-dim) for gene perturbation
            cell_line: Metadata label for the supplied profile; it does not load data
            custom_expression: Required real cell expression profile (10085 genes)
            control_expression: Required matched L1000 control profile (978 genes)
                when model_type='l1000'; unused in bulk_rnaseq mode
            return_latent: Whether to return latent representations

        Returns:
            Dictionary with predictions:
                - expression: Predicted expression values (978 or 10085 depending on model_type)
                - delta: Fold changes vs control
                - genes: Gene names
                - top_genes: Top differentially expressed genes
                - latent: Latent representation (if return_latent=True)
        """
        with torch.no_grad():
            rna_profile = self._validated_profile(custom_expression, 10085, "custom_expression")
            rna_seq = torch.from_numpy(rna_profile).unsqueeze(0).to(self.device)
            control = None
            if self.model_type == 'l1000':
                control_profile = self._validated_profile(control_expression, 978, "control_expression")
                control = torch.from_numpy(control_profile).unsqueeze(0).to(self.device)

            if self.perturbation == 'drug':
                if drug_smiles is None:
                    raise ValueError("drug_smiles required for drug perturbation")

                # Tokenize drug SMILES
                encoded = self.tokenizer(
                    drug_smiles,
                    return_tensors='pt',
                    max_length=160,
                    padding='max_length',
                    truncation=True
                )
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                if self.model_type == 'bulk_rnaseq':
                    # RNADrugResponsePredictor returns (output, delta_z)
                    x_pred, delta_z = self.predictor(rna_seq, input_ids, attention_mask)
                    z_base = None
                else:
                    # JointPerturbationPredictor returns (x_pred, z_pred, delta_z, z_base)
                    x_pred, z_pred, delta_z, z_base = self.predictor(rna_seq, control, input_ids, attention_mask)

            else:  # gene perturbation (shrna, oe, xpr)
                if gene_embedding_ppi is None or gene_embedding_seq is None:
                    raise ValueError("gene_embedding_ppi and gene_embedding_seq required for gene perturbation")

                sh_emb_PPI = torch.from_numpy(gene_embedding_ppi).float().unsqueeze(0).to(self.device)
                sh_emb_seq = torch.from_numpy(gene_embedding_seq).float().unsqueeze(0).to(self.device)

                if self.model_type == 'bulk_rnaseq':
                    # RNAGeneResponsePredictor returns (output, delta_z)
                    x_pred, delta_z = self.predictor(rna_seq, sh_emb_PPI, sh_emb_seq)
                    z_base = None
                else:
                    # JointPerturbationPredictor_sh returns (x_pred, z_pred, delta_z, z_base)
                    x_pred, z_pred, delta_z, z_base = self.predictor(rna_seq, control, sh_emb_PPI, sh_emb_seq)

            # Process results
            expression = x_pred.cpu().numpy()[0]

            # For bulk_rnaseq, control is the input rna_seq
            if self.model_type == 'bulk_rnaseq':
                control_np = rna_seq.cpu().numpy()[0]
            else:
                control_np = control.cpu().numpy()[0]

            delta = expression - control_np

            # Get gene names
            num_genes = len(expression)
            genes = [f"gene_{i}" for i in range(num_genes)]

            # Find top DEGs
            top_indices = np.argsort(np.abs(delta))[-20:][::-1]
            top_genes = [
                {
                    'gene': genes[i],
                    'expression': float(expression[i]),
                    'fold_change': float(delta[i]),
                    'direction': 'up' if delta[i] > 0 else 'down'
                }
                for i in top_indices
            ]

            result = {
                'expression': expression,
                'delta': delta,
                'genes': genes,
                'top_genes': top_genes,
                'model_type': self.model_type,
                'perturbation': self.perturbation,
                'output_dim': self.output_dim,
                'cell_line': cell_line,
                'input_status': (
                    'real_custom_expression_and_l1000_control'
                    if self.model_type == 'l1000'
                    else 'real_custom_expression'
                ),
            }

            if return_latent:
                result['latent'] = {
                    'delta_z': delta_z.cpu().numpy()[0] if delta_z is not None else None,
                    'z_base': z_base.cpu().numpy()[0] if z_base is not None else None
                }

            return result

    def predict_batch(self,
                     drug_smiles_list: List[str],
                     cell_line: str = 'MCF7',
                     custom_expression: Optional[np.ndarray] = None,
                     control_expression: Optional[np.ndarray] = None,
                     batch_size: int = 32) -> List[Dict]:
        """
        Predict transcriptome changes for multiple drugs

        Args:
            drug_smiles_list: List of SMILES strings
            cell_line: Cell line name
            custom_expression: Required real expression profile (10085 genes)
            control_expression: Required matched control (978 genes) in L1000 mode
            batch_size: Batch size for inference

        Returns:
            List of prediction dictionaries
        """
        results = []
        for i in range(0, len(drug_smiles_list), batch_size):
            batch_smiles = drug_smiles_list[i:i + batch_size]
            for smiles in batch_smiles:
                try:
                    result = self.predict(
                        drug_smiles=smiles,
                        cell_line=cell_line,
                        custom_expression=custom_expression,
                        control_expression=control_expression,
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Failed to predict for {smiles}: {e}")
                    results.append({'error': str(e), 'smiles': smiles})

        return results

    def predict_gene(self,
                    gene: str,
                    custom_expression: Optional[np.ndarray] = None,
                    control_expression: Optional[np.ndarray] = None,
                    return_latent: bool = False) -> Dict:
        """
        Predict transcriptome changes from gene perturbation using gene name.

        This is a convenience method that automatically loads gene embeddings.

        Args:
            gene: Gene symbol (e.g., 'TP53') or ENSG ID (e.g., 'ENSG00000141510')
            custom_expression: Required real expression profile array (10085 genes)
            control_expression: Required matched control array (978 genes) in L1000 mode
            return_latent: Whether to return latent representations

        Returns:
            Dictionary with predictions (same as predict())
        """
        if self.perturbation == 'drug':
            raise ValueError("predict_gene() is for gene perturbation models. Use predict() with drug_smiles for drug models.")

        # Load gene embedding loader
        from gene_embedding_loader import GeneEmbeddingLoader, COMMON_GENES

        loader = GeneEmbeddingLoader(self.model_dir / "gene_embeddings")

        # Convert symbol to ENSG ID if needed
        if gene.startswith('ENSG'):
            ensg_id = gene
        elif gene in COMMON_GENES:
            ensg_id = COMMON_GENES[gene]
        else:
            raise ValueError(f"Unknown gene: {gene}. Please provide ENSG ID or use a common gene symbol.")

        # Get embeddings
        ppi_emb, seq_emb = loader.get_embeddings(ensg_id)

        # Call predict with embeddings
        return self.predict(
            gene_embedding_ppi=ppi_emb,
            gene_embedding_seq=seq_emb,
            custom_expression=custom_expression,
            control_expression=control_expression,
            return_latent=return_latent
        )

    def expand_l1000_to_full(self,
                            l1000_expression: np.ndarray,
                            clip_range: tuple = (0, 15)) -> Dict:
        """
        Expand L1000 978-gene expression to full 12328-gene expression.

        Uses the official LINCS inference method to predict expression of
        ~11350 additional genes from 978 landmark genes.

        This is useful when you have L1000 model output (978 genes) and want
        to expand it to a more complete transcriptome.

        Args:
            l1000_expression: Expression values for 978 L1000 genes.
                             Shape: (978,) for single sample.
            clip_range: (min, max) to clip inferred values. Default (0, 15).

        Returns:
            Dictionary with:
                - expression: Full expression values (12328 genes)
                - landmark_genes: List of 978 landmark gene names
                - inferred_genes: List of 11350 inferred gene names
                - all_genes: List of all 12328 gene names
        """
        from l1000_inference_tool import L1000GeneInference

        # Initialize inference tool
        inferrer = L1000GeneInference(
            data_dir=self.model_dir / "l1000_inference",
            input_type='symbol'
        )

        # Perform inference
        full_expression = inferrer.infer(l1000_expression, clip_range=clip_range)

        return {
            'expression': full_expression,
            'landmark_genes': inferrer.get_landmark_genes(),
            'inferred_genes': inferrer.get_inferred_genes(),
            'all_genes': inferrer.get_all_genes(),
            'n_landmark': len(inferrer.get_landmark_genes()),
            'n_inferred': len(inferrer.get_inferred_genes()),
            'n_total': len(inferrer.get_all_genes())
        }


def expand_l1000_expression(l1000_expression: np.ndarray) -> Dict:
    """
    Convenience function to expand L1000 expression to full transcriptome.

    Args:
        l1000_expression: Expression values for 978 L1000 genes.

    Returns:
        Dictionary with full expression (12328 genes) and gene lists.
    """
    from l1000_inference_tool import L1000GeneInference

    inferrer = L1000GeneInference(input_type='symbol')
    full_expression = inferrer.infer(l1000_expression)

    return {
        'expression': full_expression,
        'landmark_genes': inferrer.get_landmark_genes(),
        'inferred_genes': inferrer.get_inferred_genes(),
        'all_genes': inferrer.get_all_genes(),
        'n_total': len(inferrer.get_all_genes())
    }


# Convenience functions
def predict_drug_response(drug_smiles: str,
                         cell_line: str = 'MCF7',
                         model_type: str = 'l1000',
                         custom_expression: Optional[np.ndarray] = None,
                         control_expression: Optional[np.ndarray] = None) -> Dict:
    """
    Quick prediction for drug response

    Args:
        drug_smiles: Drug SMILES string
        cell_line: Cell line name
        model_type: 'l1000' (978 genes) or 'bulk_rnaseq' (10085 genes)
        custom_expression: Required real expression profile (10085 genes)
        control_expression: Required matched control (978 genes) in L1000 mode

    Returns:
        Prediction dictionary
    """
    predictor = TranscriptomePredictor(model_type=model_type, perturbation='drug')
    return predictor.predict(
        drug_smiles=drug_smiles,
        cell_line=cell_line,
        custom_expression=custom_expression,
        control_expression=control_expression,
    )


def predict_gene_knockdown(gene: str,
                          model_type: str = 'bulk_rnaseq',
                          custom_expression: Optional[np.ndarray] = None,
                          control_expression: Optional[np.ndarray] = None) -> Dict:
    """
    Quick prediction for gene knockdown (shRNA)

    Args:
        gene: Gene symbol (e.g., 'TP53') or ENSG ID
        model_type: 'l1000' (978 genes) or 'bulk_rnaseq' (10085 genes)
        custom_expression: Required real expression profile (10085 genes)
        control_expression: Required matched control (978 genes) in L1000 mode

    Returns:
        Prediction dictionary
    """
    predictor = TranscriptomePredictor(model_type=model_type, perturbation='shrna')
    return predictor.predict_gene(
        gene=gene,
        custom_expression=custom_expression,
        control_expression=control_expression,
    )


def predict_gene_overexpression(gene: str,
                               model_type: str = 'bulk_rnaseq',
                               custom_expression: Optional[np.ndarray] = None,
                               control_expression: Optional[np.ndarray] = None) -> Dict:
    """
    Quick prediction for gene overexpression

    Args:
        gene: Gene symbol (e.g., 'TP53') or ENSG ID
        model_type: 'l1000' (978 genes) or 'bulk_rnaseq' (10085 genes)
        custom_expression: Required real expression profile (10085 genes)
        control_expression: Required matched control (978 genes) in L1000 mode

    Returns:
        Prediction dictionary
    """
    predictor = TranscriptomePredictor(model_type=model_type, perturbation='oe')
    return predictor.predict_gene(
        gene=gene,
        custom_expression=custom_expression,
        control_expression=control_expression,
    )


def predict_gene_knockout(gene: str,
                         model_type: str = 'bulk_rnaseq',
                         custom_expression: Optional[np.ndarray] = None,
                         control_expression: Optional[np.ndarray] = None) -> Dict:
    """
    Quick prediction for gene knockout (CRISPR)

    Args:
        gene: Gene symbol (e.g., 'TP53') or ENSG ID
        model_type: 'l1000' (978 genes) or 'bulk_rnaseq' (10085 genes)
        custom_expression: Required real expression profile (10085 genes)
        control_expression: Required matched control (978 genes) in L1000 mode

    Returns:
        Prediction dictionary
    """
    predictor = TranscriptomePredictor(model_type=model_type, perturbation='xpr')
    return predictor.predict_gene(
        gene=gene,
        custom_expression=custom_expression,
        control_expression=control_expression,
    )


if __name__ == "__main__":
    example = Path(__file__).resolve().parents[2] / "examples" / "api_context_examples.npz"
    if not example.exists():
        raise SystemExit(
            f"Required real example profile not found: {example}\n"
            "Run the repository model downloader/patcher or supply your own documented profiles."
        )
    context = np.load(example, allow_pickle=False)
    cell_line = str(context["cell_id"][0])
    rna_profile = context["rna"][0]
    l1000_control = context["control"][0]

    print("Transcriptome Prediction - deterministic real-input example")
    print("=" * 60)
    predictor_l1000 = TranscriptomePredictor(model_type='l1000', perturbation='drug')
    result_l1000 = predictor_l1000.predict(
        drug_smiles='CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
        cell_line=cell_line,
        custom_expression=rna_profile,
        control_expression=l1000_control,
    )
    print(f"Output dimension: {result_l1000['expression'].shape}")
    print("Top 3 DEGs:")
    for gene in result_l1000['top_genes'][:3]:
        print(f"  {gene['gene']}: {gene['fold_change']:.3f} ({gene['direction']})")
