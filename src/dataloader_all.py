import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm
import h5py
from pathlib import Path
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

from aethercell.data import safe_load_mapping

class CustomHDF5Dataset(Dataset):
    def __init__(self, h5_file):
        """
        Initialize custom dataset
        :param h5_file: path to h5 file
        """
        # Read h5 file using h5py
        with h5py.File(h5_file, 'r') as f:
            # Get all dataset names in the file
            self.dataset_names = list(f.keys())
            # Get dataset contents and convert to numpy arrays
            self.data = {name: f[name][:] for name in self.dataset_names}

        # Convert data to torch tensors
        self.tensor_data = {name: torch.tensor(data, dtype=torch.float32) for name, data in self.data.items()}

    def __len__(self):
        """
        Return the size of dataset (number of datasets)
        """
        return len(self.dataset_names)

    def __getitem__(self, idx):
        """
        Return data and corresponding column name by index
        :param idx: index
        :return: (data, column_name)
        """
        # Get data
        dataset_name = self.dataset_names[idx]
        data = self.tensor_data[dataset_name]
        return data, dataset_name


class RNADataset(Dataset):
    def __init__(self, RNA_parquet_path: str):
        df = pd.read_parquet(RNA_parquet_path)     # (G, C)
        self.cells = list(df.columns)              # ['A549', 'MCF7', ...]
        self.rna = df.to_numpy(dtype=np.float32, copy=False)  # (G, C)
    def __len__(self):
        return len(self.cells)                     # C

    def __getitem__(self, idx: int):
        cell = self.cells[idx]
        vec = self.rna[:, idx]                     # (G,), float32
        return {
            "cell": cell,
            "rna": torch.from_numpy(vec)           # Tensor(G,)
        }

class DDPDataset(Dataset):
    def __init__(self, meta_csv,RNA_parquet_path,
                 drug_input_ids_npy, drug_attention_mask_npy,
                 drug_idx_map_path):
        self.meta =  pd.read_csv(meta_csv)
        df_RNA = pd.read_parquet(RNA_parquet_path)  # Assume rows are genes, columns are cell_iname, shape (G, C)
        #    G = number of genes, C = number of different cell_iname
        self.RNA_arr = df_RNA.values.astype(np.float32)   # shape (G, C)
        self.cell_list = list(df_RNA.columns)             # ['A549', 'MCF7', ...]
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.cell_list)}
        arr_input_ids = np.load(drug_input_ids_npy, mmap_mode='r')     # (N_drug, 160)
        arr_attention = np.load(drug_attention_mask_npy, mmap_mode='r')# (N_drug, 160)
        self.drug_input_ids = torch.from_numpy(arr_input_ids.astype(np.int64))        # (N_drug, 160)
        self.drug_attention_mask = torch.from_numpy(arr_attention.astype(np.int64))   # (N_drug, 160)
        self.drug_idx_map = safe_load_mapping(drug_idx_map_path)  # { pert_id: idx }
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, index):
        cell_name = self.meta.loc[index, "ccle_name"]
        col_idx = self.cell_to_idx[cell_name]
        RNAdata = self.RNA_arr[:, col_idx]
        label = self.meta.loc[index, "label"]
        drug_name = self.meta.loc[index, "broad_id"]
        d_idx = self.drug_idx_map[drug_name]       # int
        input_ids = self.drug_input_ids[d_idx]   # (160,), int64
        attention_mask = self.drug_attention_mask[d_idx]  # (160,), int64
        return {'label': label,'cell_data': RNAdata,'input_ids': input_ids,'attention_mask': attention_mask}

class DDPDataset_ic50(Dataset):
    def __init__(self, meta_csv,RNA_parquet_path,
                 drug_input_ids_npy, drug_attention_mask_npy,
                 drug_idx_map_path):
        self.meta =  pd.read_csv(meta_csv)
        df_RNA = pd.read_parquet(RNA_parquet_path)  # Assume rows are genes, columns are cell_iname, shape (G, C)
        #    G = number of genes, C = number of different cell_iname
        self.RNA_arr = df_RNA.values.astype(np.float32)   # shape (G, C)
        self.cell_list = list(df_RNA.columns)             # ['A549', 'MCF7', ...]
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.cell_list)}
        arr_input_ids = np.load(drug_input_ids_npy, mmap_mode='r')     # (N_drug, 160)
        arr_attention = np.load(drug_attention_mask_npy, mmap_mode='r')# (N_drug, 160)
        self.drug_input_ids = torch.from_numpy(arr_input_ids.astype(np.int64))        # (N_drug, 160)
        self.drug_attention_mask = torch.from_numpy(arr_attention.astype(np.int64))   # (N_drug, 160)
        self.drug_idx_map = safe_load_mapping(drug_idx_map_path)  # { pert_id: idx }
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, index):
        cell_name = self.meta.loc[index, "CELL_LINE_NAME"]
        col_idx = self.cell_to_idx[cell_name]
        RNAdata = self.RNA_arr[:, col_idx]
        label = self.meta.loc[index, "LN_IC50"]
        drug_name = self.meta.loc[index, "DRUG_NAME"]
        # data_type = self.meta.loc[index, "data_type"]
        d_idx = self.drug_idx_map[drug_name]       # int
        input_ids = self.drug_input_ids[d_idx]   # (160,), int64
        attention_mask = self.drug_attention_mask[d_idx]  # (160,), int64
        # return {"data_type":data_type,'label': label,'cell_data': RNAdata,'input_ids': input_ids,'attention_mask': attention_mask}
        return {'label': label,'cell_data': RNAdata,'input_ids': input_ids,'attention_mask': attention_mask}


class PretrainDataset(Dataset):
    """
    One training set + multiple test sets share one RNA/LINCS cache.
    Returns: (rna, l1000, control, is_perturb, RNAseq_name, L_name)
    """
    PERTURB_SET = {"trt_cp", "trt_sh", "trt_xpr", "trt_oe"}  # Others are treated as control

    def __init__(self, meta_csv: str, rna_dict: dict, lincs_dict: dict, drop_missing=True):
        self.meta = pd.read_csv(meta_csv)
        self.rna_dict = rna_dict
        self.lincs_dict = lincs_dict

        # Pre-fetch needed columns as numpy arrays to avoid slow DataFrame queries in __getitem__
        needed_cols = ["cell_iname", "sample_id", "pert_type", "representative_control"]
        for c in needed_cols:
            if c not in self.meta.columns:
                raise KeyError(f"{meta_csv} missing required column: {c}")
        self.cell_iname = self.meta["cell_iname"].astype(str).values
        self.sample_id = self.meta["sample_id"].astype(str).values
        self.pert_type = self.meta["pert_type"].astype(str).values
        # Control may have missing values, use empty string as placeholder
        self.ctrl_id = self.meta["representative_control"].fillna("").astype(str).values
        # Dimension check and missing data filtering
        # Use length of any LINCS vector as dimension (e.g., 978)
        any_vec = next(iter(lincs_dict.values()))
        self.gene_dim = int(any_vec.shape[0])
        if drop_missing:
            keep = np.ones(len(self.meta), dtype=bool)

            # Ensure sample_id exists in LINCS
            miss_sample = [i for i, sid in enumerate(self.sample_id) if sid not in lincs_dict]
            if len(miss_sample) > 0:
                keep[miss_sample] = False

            # Ensure RNA cell line exists (could use zero vector, but choose to filter for stability)
            miss_cell = [i for i, cname in enumerate(self.cell_iname) if cname not in rna_dict]
            if len(miss_cell) > 0:
                keep[miss_cell] = False
            # Apply filter
            self.cell_iname = self.cell_iname[keep]
            self.sample_id = self.sample_id[keep]
            self.pert_type = self.pert_type[keep]
            self.ctrl_id   = self.ctrl_id[keep]

        # Statistics (optional)
        # print(f"[{os.path.basename(meta_csv)}] kept {len(self)} samples")
    def __len__(self):
        return len(self.sample_id)
    def __getitem__(self, index):
        cname = self.cell_iname[index]
        sid   = self.sample_id[index]
        ptype = self.pert_type[index]
        cid   = self.ctrl_id[index]  # May be empty string
        # RNAseq (if missing, use zero vector as fallback to ensure batch concatenation)
        rna_vec = self.rna_dict.get(cname, None)
        if rna_vec is None:
            rna = torch.zeros(self.gene_dim, dtype=torch.float32)
        else:
            rna = torch.from_numpy(rna_vec)
        # L1000 sample
        l_vec = self.lincs_dict.get(sid, None)
        if l_vec is None:
            # Should be filtered in theory, but fallback here
            l1000 = torch.zeros(self.gene_dim, dtype=torch.float32)
        else:
            l1000 = torch.from_numpy(l_vec)
        # Whether perturbation
        is_perturb = 1 if ptype in self.PERTURB_SET else 0
        # Control (only try to fetch corresponding control if perturbed)
        if is_perturb == 1 and cid and (cid in self.lincs_dict):
            c_vec = self.lincs_dict[cid]
            control = torch.from_numpy(c_vec)
        else:
            control = torch.zeros_like(l1000)
        return rna, l1000, control, is_perturb, cname, sid
    

class PretrainDataset_control(Dataset):
    def __init__(self, meta_csv, LINCS_table):
        self.meta = pd.read_csv(meta_csv)
        table_path = Path(LINCS_table)
        if table_path.suffix.lower() == ".parquet":
            lincs_df = pd.read_parquet(table_path)
        elif table_path.suffix.lower() in {".csv", ".tsv"}:
            separator = "\t" if table_path.suffix.lower() == ".tsv" else ","
            lincs_df = pd.read_csv(table_path, sep=separator, index_col=0)
        else:
            raise ValueError(
                "LINCS_table must be a safe .parquet, .csv, or .tsv file. "
                "Legacy DataFrame pickle loading was removed because pickle can execute arbitrary code."
            )
        self.LINCS_dict = {
            str(idx): np.asarray(lincs_df.loc[idx], dtype=np.float32)
            for idx in lincs_df.index
        }
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, index):
        # Get meta info
        row = self.meta.loc[index]
        RNAseq_name = row["cell_iname"]
        L_name = row["sample_id"]
        # L1000
        l1000 = torch.from_numpy(self.LINCS_dict[L_name])
        return l1000, RNAseq_name,L_name



class PredictorDatasetDP2_sh(Dataset):
    def __init__(self,
                 meta_csv,
                 L1000_exp_npy, L1000_ctrl_npy,
                 exp_idx_map_path, ctrl_idx_map_path,
                 RNA_parquet_path,
                 sh_embed_PPI_csv,
                 sh_seq_emb_npy,
                 sh_seq_emb_pkl):
        # 1. Read meta.csv (pandas)
        self.meta = pd.read_csv(meta_csv, engine="python")
        # 2. Read L1000 (directly convert to float32)
        self.L1000_exp = np.load(L1000_exp_npy, mmap_mode='r').astype(np.float32)   # (N_exp, 978)
        self.L1000_ctrl = np.load(L1000_ctrl_npy, mmap_mode='r').astype(np.float32) # (N_ctrl, 978)
        # 3. Read index mapping (directly use dict lookup)
        self.exp_idx_map = safe_load_mapping(exp_idx_map_path)   # { sample_id: row_in_L1000_exp }
        self.ctrl_idx_map = safe_load_mapping(ctrl_idx_map_path) # { sample_id: row_in_L1000_ctrl }
        # 4. Read RNAseq parquet -> convert to NumPy first and cache column name -> index
        df_RNA = pd.read_parquet(RNA_parquet_path)  # Assume rows are genes, columns are cell_iname, shape (G, C)
        #    G = number of genes, C = number of different cell_iname
        self.RNA_arr = df_RNA.values.astype(np.float32)   # shape (G, C)
        self.cell_list = list(df_RNA.columns)             # ['A549', 'MCF7', ...]
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.cell_list)}
        del df_RNA
        df_sh_PPI = pd.read_csv(sh_embed_PPI_csv, engine="python")      # shape (dim, num_sh)
        emb_cols = [c for c in df_sh_PPI.columns if c not in {"gene", "pert_id"}]
        df_sh_PPI = df_sh_PPI.set_index("gene")
        df_sh_PPI = df_sh_PPI[emb_cols].astype(np.float32)
        self.sh_PPI_arr = df_sh_PPI.to_numpy(copy=False)          # shape: (num_genes, dim)
        self.sh_PPI_ids = df_sh_PPI.index.astype(str).tolist()    # ['GENE1', 'GENE2', ...]
        self.sh_PPI_id2row = {gid: i for i, gid in enumerate(self.sh_PPI_ids)}
        self.sh_PPI_embedding_dim = self.sh_PPI_arr.shape[1]
        del df_sh_PPI
        self.sh_seq  = np.load(sh_seq_emb_npy)
        self.sh_seq_id2idx = safe_load_mapping(sh_seq_emb_pkl)
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]            # Pandas Series
        # 1) L1000 data for control / label
        ctrl_id = row["representative_control"]
        ctrl_rownum = self.ctrl_idx_map[ctrl_id]
        control_vec = self.L1000_ctrl[ctrl_rownum]   # np.float32, shape (978,)

        label_id = row["sample_id"]
        det_plate = row["det_plate"]
        exp_rownum = self.exp_idx_map[label_id]
        label_vec = self.L1000_exp[exp_rownum]       # np.float32, shape (978,)

        # 2) RNAseq: directly index NumPy by cell_iname
        cell = row["cell_iname"]
        col_idx = self.cell_to_idx[cell]
        rna_vec = self.RNA_arr[:, col_idx]           # shape (G,), np.float32

        # 3) Drug / sh embeddings
        pert_type = row["pert_type"]
        pert_id = row["gene_ensg"]
        PPI_emb = self.sh_PPI_arr[self.sh_PPI_id2row[pert_id], :]
        seq_idx = self.sh_seq_id2idx[pert_id]
        seq_vec = self.sh_seq[seq_idx, :]  # dtype float16/32
        if seq_vec.dtype != np.float32:
            seq_vec = seq_vec.astype(np.float32, copy=False)
        return {
            "sample_id": label_id,
            "pert_type": pert_type,
            "cell_id" : cell,
            "det_plate" : det_plate,
            "control": torch.from_numpy(control_vec),       # (978,), float32
            "label":   torch.from_numpy(label_vec),         # (978,), float32
            "rna":     torch.from_numpy(rna_vec),           # (G,), float32
            "control_id": ctrl_id,
            "pert_id" : pert_id,             # (160,), int64
            "sh_PPI_embedding": torch.from_numpy(PPI_emb.astype(np.float32)),
            "sh_seq_embedding": torch.from_numpy(seq_vec)}
    

class PredictorDatasetDP2(Dataset):
    def __init__(self,
                 meta_csv,
                 L1000_exp_npy, L1000_ctrl_npy,
                 exp_idx_map_path, ctrl_idx_map_path,
                 RNA_parquet_path,
                 drug_input_ids_npy, drug_attention_mask_npy,
                 drug_idx_map_path,):
                #  sh_embed_PPI_csv, sh_embed_seq_csv):
        # 1. Read meta.csv (pandas)
        self.meta = pd.read_csv(meta_csv, engine="python")
        #    meta contains sample_id, representative_control, pert_id, pert_type, cell_iname

        # 2. Read L1000 (directly convert to float32)
        self.L1000_exp = np.load(L1000_exp_npy, mmap_mode='r').astype(np.float32)   # (N_exp, 978)
        self.L1000_ctrl = np.load(L1000_ctrl_npy, mmap_mode='r').astype(np.float32) # (N_ctrl, 978)

        # 3. Read index mapping (directly use dict lookup)
        self.exp_idx_map = safe_load_mapping(exp_idx_map_path)   # { sample_id: row_in_L1000_exp }
        self.ctrl_idx_map = safe_load_mapping(ctrl_idx_map_path) # { sample_id: row_in_L1000_ctrl }

        # 4. Read RNAseq parquet -> convert to NumPy first and cache column name -> index
        df_RNA = pd.read_parquet(RNA_parquet_path)  # Assume rows are genes, columns are cell_iname, shape (G, C)
        #    G = number of genes, C = number of different cell_iname
        self.RNA_arr = df_RNA.values.astype(np.float32)   # shape (G, C)
        self.cell_list = list(df_RNA.columns)             # ['A549', 'MCF7', ...]
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.cell_list)}

        # 5. Read Drug Tokenize results -> mmap or load as Torch Tensor in one go
        arr_input_ids = np.load(drug_input_ids_npy, mmap_mode='r')     # (N_drug, 160)
        arr_attention = np.load(drug_attention_mask_npy, mmap_mode='r')# (N_drug, 160)
        # If GPU memory allows, directly convert to Tensor and cache in memory for direct indexing; otherwise keep as NumPy but ensure np.int64 / np.int32
        self.drug_input_ids = torch.from_numpy(arr_input_ids.astype(np.int64))        # (N_drug, 160)
        self.drug_attention_mask = torch.from_numpy(arr_attention.astype(np.int64))   # (N_drug, 160)
        self.drug_idx_map = safe_load_mapping(drug_idx_map_path)  # { pert_id: idx }

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]            # Pandas Series
        # 1) L1000 data for control / label
        ctrl_id = row["representative_control"]
        ctrl_rownum = self.ctrl_idx_map[ctrl_id]
        control_vec = self.L1000_ctrl[ctrl_rownum]   # np.float32, shape (978,)

        label_id = row["sample_id"]
        det_plate = label_id.split(':')[0]
        exp_rownum = self.exp_idx_map[label_id]
        label_vec = self.L1000_exp[exp_rownum]       # np.float32, shape (978,)

        # 2) RNAseq: directly index NumPy by cell_iname
        cell = row["cell_iname"]
        col_idx = self.cell_to_idx[cell]
        rna_vec = self.RNA_arr[:, col_idx]           # shape (G,), np.float32

        # 3) Drug / sh embeddings
        pert_type = row["pert_type"]
        pert_id = row["pert_id"]

        # 3.1. Pre-initialize them with zero vectors
        input_ids = torch.zeros((160,), dtype=torch.int64)
        attention_mask = torch.zeros((160,), dtype=torch.int64)
        # sh_PPI_embedding = torch.zeros((self.sh_PPI_embedding_dim,), dtype=torch.float32)
        # sh_seq_embedding = torch.zeros((self.sh_seq_embedding_dim,), dtype=torch.float32)

        if pert_type == "trt_cp":
            d_idx = self.drug_idx_map[pert_id]       # int
            input_ids = self.drug_input_ids[d_idx]   # (160,), int64
            attention_mask = self.drug_attention_mask[d_idx]  # (160,), int64

        elif pert_type == "trt_sh":
            # Directly use NumPy slicing + torch.from_numpy
            col = self.sh_PPI_id2col[pert_id]
            PPI_emb = self.sh_PPI_arr[:, col]         # (dim,), np.float32
            sh_PPI_embedding = torch.from_numpy(PPI_emb)

            col2 = self.sh_seq_id2col[pert_id]
            seq_emb = self.sh_seq_arr[:, col2]        # (dim,), np.float32
            sh_seq_embedding = torch.from_numpy(seq_emb)

        return {
            "sample_id": label_id,
            "pert_type": pert_type,
            "control": torch.from_numpy(control_vec),       # (978,), float32
            "label":   torch.from_numpy(label_vec),         # (978,), float32
            "rna":     torch.from_numpy(rna_vec),           # (G,), float32
            "input_ids":      input_ids,                    # (160,), int64
            "attention_mask": attention_mask,
            "control_id": ctrl_id,
            "pert_id" : pert_id,             # (160,), int64
            "cell_id" : cell,
            "det_plate" : det_plate
        }

class PredictorDatasetDP2_sh_i(Dataset):
    def __init__(self,
                 meta_csv,L1000_ctrl_npy,ctrl_idx_map_path,
                 RNA_parquet_path,
                 sh_embed_PPI_csv,
                 sh_seq_emb_npy,
                 sh_seq_emb_pkl):
        # 1. Read meta.csv (pandas)
        self.meta = pd.read_csv(meta_csv, engine="python")
        # 2. Read L1000 (directly convert to float32)
        self.L1000_ctrl = np.load(L1000_ctrl_npy, mmap_mode='r').astype(np.float32) # (N_ctrl, 978)
        # 3. Read index mapping (directly use dict lookup)
        self.ctrl_idx_map = safe_load_mapping(ctrl_idx_map_path) # { sample_id: row_in_L1000_ctrl }
        # 4. Read RNAseq parquet -> convert to NumPy first and cache column name -> index
        df_RNA = pd.read_parquet(RNA_parquet_path)  # Assume rows are genes, columns are cell_iname, shape (G, C)
        #    G = number of genes, C = number of different cell_iname
        self.RNA_arr = df_RNA.values.astype(np.float32)   # shape (G, C)
        self.cell_list = list(df_RNA.columns)             # ['A549', 'MCF7', ...]
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.cell_list)}
        del df_RNA
        df_sh_PPI = pd.read_csv(sh_embed_PPI_csv, engine="python")      # shape (dim, num_sh)
        emb_cols = [c for c in df_sh_PPI.columns if c not in {"gene", "pert_id"}]
        df_sh_PPI = df_sh_PPI.set_index("gene")
        df_sh_PPI = df_sh_PPI[emb_cols].astype(np.float32)
        self.sh_PPI_arr = df_sh_PPI.to_numpy(copy=False)          # shape: (num_genes, dim)
        self.sh_PPI_ids = df_sh_PPI.index.astype(str).tolist()    # ['GENE1', 'GENE2', ...]
        self.sh_PPI_id2row = {gid: i for i, gid in enumerate(self.sh_PPI_ids)}
        self.sh_PPI_embedding_dim = self.sh_PPI_arr.shape[1]
        del df_sh_PPI
        self.sh_seq  = np.load(sh_seq_emb_npy)
        self.sh_seq_id2idx = safe_load_mapping(sh_seq_emb_pkl)
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]            # Pandas Series
        # 1) L1000 data for control / label
        ctrl_id = row["representative_control"]
        ctrl_rownum = self.ctrl_idx_map[ctrl_id]
        control_vec = self.L1000_ctrl[ctrl_rownum]   # np.float32, shape (978,)
        label_id = row["sample_id"]

        # 2) RNAseq: directly index NumPy by cell_iname
        cell = row["cell_iname"]
        col_idx = self.cell_to_idx[cell]
        rna_vec = self.RNA_arr[:, col_idx]           # shape (G,), np.float32

        # 3) Drug / sh embeddings
        pert_type = row["pert_type"]
        pert_id = row["gene_ensg"]
        PPI_emb = self.sh_PPI_arr[self.sh_PPI_id2row[pert_id], :]
        seq_idx = self.sh_seq_id2idx[pert_id]
        seq_vec = self.sh_seq[seq_idx, :]  # dtype float16/32
        if seq_vec.dtype != np.float32:
            seq_vec = seq_vec.astype(np.float32, copy=False)
        return {
            "sample_id": label_id,
            "pert_type": pert_type,
            "control": torch.from_numpy(control_vec),       # (978,), float32
            "rna":     torch.from_numpy(rna_vec),           # (G,), float32
            "control_id": ctrl_id,
            "pert_id" : pert_id,             # (160,), int64
            "sh_PPI_embedding": torch.from_numpy(PPI_emb.astype(np.float32)),
            "sh_seq_embedding": torch.from_numpy(seq_vec)}
    


class PredictorDatasetDP2_i(Dataset):
    def __init__(self,
                 meta_csv,
                 L1000_ctrl_npy,ctrl_idx_map_path,
                 RNA_parquet_path,
                 drug_input_ids_npy, drug_attention_mask_npy,
                 drug_idx_map_path,):
        self.meta = pd.read_csv(meta_csv, engine="python")
        self.L1000_ctrl = np.load(L1000_ctrl_npy, mmap_mode='r').astype(np.float32) # (N_ctrl, 978)
        self.ctrl_idx_map = safe_load_mapping(ctrl_idx_map_path) # { sample_id: row_in_L1000_ctrl }
        df_RNA = pd.read_parquet(RNA_parquet_path)  # Assume rows are genes, columns are cell_iname, shape (G, C)
        #    G = number of genes, C = number of different cell_iname
        self.RNA_arr = df_RNA.values.astype(np.float32)   # shape (G, C)
        self.cell_list = list(df_RNA.columns)             # ['A549', 'MCF7', ...]
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.cell_list)}

        # 5. Read Drug Tokenize results -> mmap or load as Torch Tensor in one go
        arr_input_ids = np.load(drug_input_ids_npy, mmap_mode='r')     # (N_drug, 160)
        arr_attention = np.load(drug_attention_mask_npy, mmap_mode='r')# (N_drug, 160)
        # If GPU memory allows, directly convert to Tensor and cache in memory for direct indexing; otherwise keep as NumPy but ensure np.int64 / np.int32
        self.drug_input_ids = torch.from_numpy(arr_input_ids.astype(np.int64))        # (N_drug, 160)
        self.drug_attention_mask = torch.from_numpy(arr_attention.astype(np.int64))   # (N_drug, 160)
        self.drug_idx_map = safe_load_mapping(drug_idx_map_path)  # { pert_id: idx }

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]            # Pandas Series
        # 1) L1000 data for control / label
        ctrl_id = row["representative_control"]
        ctrl_rownum = self.ctrl_idx_map[ctrl_id]
        control_vec = self.L1000_ctrl[ctrl_rownum]   # np.float32, shape (978,)
        label_id = row["sample_id"]
        # 2) RNAseq: directly index NumPy by cell_iname
        cell = row["cell_iname"]
        col_idx = self.cell_to_idx[cell]
        rna_vec = self.RNA_arr[:, col_idx]           # shape (G,), np.float32

        # 3) Drug / sh embeddings
        pert_type = row["pert_type"]
        pert_id = row["pert_id"]
        # 3.1. Pre-initialize them with zero vectors
        input_ids = torch.zeros((160,), dtype=torch.int64)
        attention_mask = torch.zeros((160,), dtype=torch.int64)
        d_idx = self.drug_idx_map[pert_id]       # int
        input_ids = self.drug_input_ids[d_idx]   # (160,), int64
        attention_mask = self.drug_attention_mask[d_idx]  # (160,), int64

        return {
            "sample_id": label_id,
            "pert_type": pert_type,
            "control": torch.from_numpy(control_vec),       # (978,), float32
            "rna":     torch.from_numpy(rna_vec),           # (G,), float32
            "input_ids":      input_ids,                    # (160,), int64
            "attention_mask": attention_mask,
            "control_id": ctrl_id,
            "pert_id" : pert_id,             # (160,), int64
        }
    


class PredictorDataset_R(Dataset):
    def __init__(self,meta_csv,control_csv,treat_csv,molformer_path,max_length=160):
        self.meta = pd.read_csv(meta_csv)
        self.control = pd.read_csv(control_csv)
        self.treat = pd.read_csv(treat_csv)
        self.tokenizer = AutoTokenizer.from_pretrained(molformer_path, trust_remote_code=True)
        self.max_length = max_length
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, index):
        name = self.meta.loc[index, "id"]
        control_name = self.meta.loc[index, "control_name"]
        control = torch.tensor(np.array(self.control[control_name], dtype=np.float32))
        treat_name = self.meta.loc[index, "treat_name"]
        treat = torch.tensor(np.array(self.treat[treat_name], dtype=np.float32))
        smiles = self.meta.loc[index, "smiles"]
        encoded = self.tokenizer(
            smiles,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        # Remove batch dimension
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {'name':name,'control': control,'treat': treat,'input_ids': input_ids,'attention_mask': attention_mask}


class PredictorDataset_R_i(Dataset):
    def __init__(self,meta_csv,control_csv,molformer_path,max_length=160):
        self.meta = pd.read_csv(meta_csv)
        self.control = pd.read_csv(control_csv)
        self.tokenizer = AutoTokenizer.from_pretrained(molformer_path, trust_remote_code=True)
        self.max_length = max_length
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, index):
        label_name = self.meta.loc[index, "treat_name"]
        control_name = self.meta.loc[index, "control_name"]
        control = torch.tensor(np.array(self.control[control_name], dtype=np.float32))
        smiles = self.meta.loc[index, "smiles"]
        encoded = self.tokenizer(
            smiles,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        # Remove batch dimension
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {'name':label_name,'control': control,'input_ids': input_ids,'attention_mask': attention_mask}
 


class PredictorDataset_R_sh_i(Dataset):
    def __init__(self,
                 meta_csv,
                 control_csv,
                 sh_embed_PPI_csv,
                 sh_seq_emb_npy,
                 sh_seq_emb_pkl):
        # 1. Read meta.csv (pandas)
        self.meta = pd.read_csv(meta_csv)
        self.control = pd.read_csv(control_csv)
        df_sh_PPI = pd.read_csv(sh_embed_PPI_csv, engine="python")      # shape (dim, num_sh)
        emb_cols = [c for c in df_sh_PPI.columns if c not in {"gene", "pert_id"}]
        df_sh_PPI = df_sh_PPI.set_index("gene")
        df_sh_PPI = df_sh_PPI[emb_cols].astype(np.float32)
        self.sh_PPI_arr = df_sh_PPI.to_numpy(copy=False)          # shape: (num_genes, dim)
        self.sh_PPI_ids = df_sh_PPI.index.astype(str).tolist()    # ['GENE1', 'GENE2', ...]
        self.sh_PPI_id2row = {gid: i for i, gid in enumerate(self.sh_PPI_ids)}
        self.sh_PPI_embedding_dim = self.sh_PPI_arr.shape[1]
        del df_sh_PPI
        self.sh_seq  = np.load(sh_seq_emb_npy)
        self.sh_seq_id2idx = safe_load_mapping(sh_seq_emb_pkl)
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]            # Pandas Series
        # 1) L1000 data for control / label
        label_id = row["treat_name"]
        ctrl_id = row["control_name"]
        pert_id = row["gene"]
        control = torch.tensor(np.array(self.control[ctrl_id], dtype=np.float32))
        PPI_emb = self.sh_PPI_arr[self.sh_PPI_id2row[pert_id], :]
        seq_idx = self.sh_seq_id2idx[pert_id]
        seq_vec = self.sh_seq[seq_idx, :]  # dtype float16/32
        if seq_vec.dtype != np.float32:
            seq_vec = seq_vec.astype(np.float32, copy=False)
        return {
            "sample_id": label_id,
            "control_id": ctrl_id,
            "control":control,
            "pert_id" : pert_id,             # (160,), int64
            "sh_PPI_embedding": torch.from_numpy(PPI_emb.astype(np.float32)),
            "sh_seq_embedding": torch.from_numpy(seq_vec)}
    
