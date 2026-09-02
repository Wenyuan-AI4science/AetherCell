"""
简化的IC50预测推理API

预测药物在癌细胞上的敏感性(IC50/AUC)。
该模型基于转录组预测模型进行微调，用于二分类任务。

Usage:
    from ic50_inference import IC50Predictor
    import numpy as np

    context = np.load('examples/api_context_examples.npz')

    predictor = IC50Predictor()
    result = predictor.predict(
        drug_smiles='CCO',
        cell_line=str(context['cell_id'][0]),
        custom_expression=context['rna'][0],
    )
    print(f"Sensitivity probability: {result['probability']:.3f}")

The API never synthesizes a cell profile: every prediction requires a real
10085-gene expression array in the documented gene order.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys

# Add parent directory to path to import transcriptome models
parent_dir = Path(__file__).parent.parent / "transcriptome_prediction"
sys.path.insert(0, str(parent_dir))

from transformers import AutoTokenizer


class IC50Predictor:
    """预测药物在癌细胞上的IC50/AUC（敏感性）"""

    def __init__(self,
                 model_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        初始化IC50预测器

        Args:
            model_dir: 模型目录（自动检测如果为None）
            device: 'cuda', 'cpu', 或 None（自动检测）
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 查找模型目录
        if model_dir is None:
            self.model_dir = Path(__file__).parent
        else:
            self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")

        # 加载模型
        print(f"🔄 加载IC50预测模型...")
        self._load_models()
        self._enforce_deterministic_eval(self.model)
        print(f"✅ 模型已加载到 {self.device}")

    @staticmethod
    def _enforce_deterministic_eval(module: nn.Module) -> None:
        """Freeze MolFormer's random-feature maps after checkpoint loading."""
        module.eval()
        for child in module.modules():
            if hasattr(child, "deterministic") and hasattr(child, "orthogonal_random_weights"):
                child.deterministic = True

    @staticmethod
    def _validated_profile(value: Optional[np.ndarray]) -> np.ndarray:
        """Validate the required real cell profile without inventing data."""
        if value is None:
            raise ValueError(
                "custom_expression is required; AetherCell never substitutes a random cell profile. "
                "Provide a real one-dimensional 10085-gene array in the documented gene order. "
                "A public-data-derived example is available at examples/api_context_examples.npz."
            )
        profile = np.asarray(value, dtype=np.float32)
        if profile.ndim != 1 or profile.shape[0] != 10085:
            raise ValueError(
                f"custom_expression must be a one-dimensional array with 10085 genes, got {profile.shape}"
            )
        if not np.isfinite(profile).all():
            raise ValueError("custom_expression contains NaN or infinite values")
        return np.ascontiguousarray(profile)

    def _load_models(self):
        """加载所有必需的模型组件"""
        # 导入模型定义
        sys.path.insert(0, str(self.model_dir))
        from ddp_predict_ic50_new import DDPPredictor

        # 导入转录组预测模型组件
        from LINCSvae import LINCSVAE
        from RNAvae import RNAVAE
        from uniperturb_drug import JointPerturbationPredictor

        # 转录组模型目录
        trans_dir = self.model_dir.parent / "transcriptome_prediction"

        print("  加载VAE模型...")
        # L1000 VAE
        l1000_vae_path = trans_dir / "L1000_vae.pt"
        LINvae_model = LINCSVAE("cpu")
        lin_ckpt = torch.load(l1000_vae_path, map_location="cpu")
        LINvae_model.load_state_dict(lin_ckpt['vae_model_state_dict'])
        encoder = LINvae_model.encoder.to(self.device)
        decoder = LINvae_model.decoder.to(self.device)

        # RNA VAE
        rna_vae_path = trans_dir / "RNA_vae.pt"
        RNAvae_model = RNAVAE("cpu")
        rna_ckpt = torch.load(rna_vae_path, map_location="cpu")
        state_dict = {k.replace('module.', ''): v for k, v in rna_ckpt['vae_model_state_dict'].items()}
        RNAvae_model.load_state_dict(state_dict)
        RNAencoder = RNAvae_model.encoder.to(self.device)

        print("  加载转录组预测模型...")
        # 加载转录组预测模型
        molformer_path = trans_dir / "molformer"
        predictor_path = trans_dir / "predictor_L_drug.pt"

        predict_model = JointPerturbationPredictor(
            encoder, decoder, RNAencoder,
            str(molformer_path), self.device
        ).to(self.device)

        predict_ckpt = torch.load(predictor_path, map_location=self.device)
        state_dict = {k.replace('module.', ''): v for k, v in predict_ckpt['model_state_dict'].items()}
        predict_model.load_state_dict(state_dict)

        print("  加载IC50预测头...")
        # 创建DDPPredictor
        ddp_model = DDPPredictor(
            pretrained_model=predict_model,
            device=self.device
        ).to(self.device)

        # 加载IC50模型权重
        ddp_ckpt_path = self.model_dir / "ddp_predictor.pt"
        ddp_ckpt = torch.load(ddp_ckpt_path, map_location=self.device)
        state_dict = {k.replace('module.', ''): v for k, v in ddp_ckpt['model_state_dict'].items()}
        ddp_model.load_state_dict(state_dict)
        ddp_model.eval()

        self.model = ddp_model

        # 加载Molformer tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(molformer_path),
            trust_remote_code=True
        )

        print("  ✅ 所有模型加载完成")

    def predict(self,
                drug_smiles: str,
                cell_line: str = 'MCF7',
                custom_expression: Optional[np.ndarray] = None) -> Dict:
        """
        预测药物在细胞上的敏感性

        Args:
            drug_smiles: 药物的SMILES字符串
            cell_line: 所提供表达谱的元数据标签；不会据此自动生成或加载数据
            custom_expression: 必需的真实表达谱数组（10085基因）

        Returns:
            字典包含预测结果:
                - probability: 敏感性概率 (0-1)
                - logit: 原始logit值
                - prediction: 二分类结果 (sensitive/resistant)
                - drug_smiles: 输入的SMILES
                - cell_line: 细胞系名称
        """
        with torch.no_grad():
            # 编码药物SMILES
            encoded = self.tokenizer(
                drug_smiles,
                return_tensors='pt',
                max_length=160,
                padding='max_length',
                truncation=True
            )
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            profile = self._validated_profile(custom_expression)
            rna_seq = torch.from_numpy(profile).unsqueeze(0).to(self.device)

            # 预测
            logit = self.model(rna_seq, input_ids, attention_mask)
            probability = torch.sigmoid(logit).item()

            # 二分类结果
            prediction = 'sensitive' if probability > 0.5 else 'resistant'

            return {
                'probability': probability,
                'logit': logit.item(),
                'prediction': prediction,
                'confidence': abs(probability - 0.5) * 2,  # 0-1的置信度
                'drug_smiles': drug_smiles,
                'cell_line': cell_line,
                'input_status': 'real_custom_expression',
            }

    def predict_batch(self,
                     drug_smiles_list: list,
                     cell_line: str = 'MCF7',
                     custom_expression: Optional[np.ndarray] = None,
                     batch_size: int = 32) -> list:
        """
        批量预测多个药物的敏感性

        Args:
            drug_smiles_list: SMILES字符串列表
            cell_line: 细胞系名称
            custom_expression: 必需的真实表达谱数组（10085基因）
            batch_size: 批处理大小

        Returns:
            预测结果列表
        """
        results = []
        for smiles in drug_smiles_list:
            try:
                result = self.predict(
                    drug_smiles=smiles,
                    cell_line=cell_line,
                    custom_expression=custom_expression,
                )
                results.append(result)
            except Exception as e:
                print(f"⚠️  预测失败 {smiles}: {e}")
                results.append({'error': str(e), 'smiles': smiles})

        return results


def predict_drug_sensitivity(drug_smiles: str,
                            cell_line: str = 'MCF7',
                            custom_expression: Optional[np.ndarray] = None) -> Dict:
    """
    快速预测药物敏感性的便捷函数

    Args:
        drug_smiles: 药物SMILES字符串
        cell_line: 细胞系名称
        custom_expression: 必需的真实表达谱数组（10085基因）

    Returns:
        预测结果字典
    """
    predictor = IC50Predictor()
    return predictor.predict(
        drug_smiles=drug_smiles,
        cell_line=cell_line,
        custom_expression=custom_expression,
    )


if __name__ == "__main__":
    example = Path(__file__).resolve().parents[2] / "examples" / "api_context_examples.npz"
    if not example.exists():
        raise SystemExit(
            f"Required real example profile not found: {example}\n"
            "Run the repository model downloader/patcher or supply your own documented profile."
        )
    context = np.load(example, allow_pickle=False)
    cell_line = str(context["cell_id"][0])

    print("IC50预测 - 真实输入、确定性示例")
    print("=" * 60)

    # 初始化预测器
    predictor = IC50Predictor()

    # 示例：预测布洛芬的敏感性
    print(f"\n📊 预测布洛芬在{cell_line}细胞上的敏感性...")
    result = predictor.predict(
        drug_smiles='CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
        cell_line=cell_line,
        custom_expression=context["rna"][0],
    )

    print(f"\n结果:")
    print(f"  敏感性概率: {result['probability']:.3f}")
    print(f"  预测: {result['prediction']}")
    print(f"  置信度: {result['confidence']:.3f}")
