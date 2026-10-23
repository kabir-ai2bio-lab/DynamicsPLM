import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import AutoTokenizer
from config import config
from itertools import cycle
import random
import h5py


def collate_batch(batch):
    collated = {}

    collated['masked_input_ids'] = pad_sequence(
        [b['masked_input_ids'] for b in batch], batch_first=True, padding_value=1)
    collated['attention_mask'] = pad_sequence(
        [b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    collated['original_input_ids'] = pad_sequence(
        [b['original_input_ids'] for b in batch], batch_first=True, padding_value=1)
    collated['mask_positions'] = pad_sequence(
        [b['mask_positions'] for b in batch], batch_first=True, padding_value=False)
    collated['res_feat'] = pad_sequence(
        [b['res_feat'] for b in batch], batch_first=True, padding_value=-1)

    max_len = collated['masked_input_ids'].shape[1]
    pair_padded = [
        F.pad(b['pair_feat'],
              (0, 0, 0, max_len - b['pair_feat'].size(0), 0, max_len - b['pair_feat'].size(0)),
              mode='constant', value=-1)
        for b in batch
    ]
    collated['pair_feat'] = torch.stack(pair_padded)

    return collated


class ProteinDatasetMLM(Dataset):
    def __init__(self, df, h5py_paths, esm2_select, max_len, mask_prob=0.15,
                 start_from_N_terminal=False):
        self.names    = list(df['name'])
        self.seqs     = list(df['modify_seq'])
        self.h5py_paths = [h5py_paths] if isinstance(h5py_paths, str) else h5py_paths
        self.h5py_files = None

        self.tokenizer            = AutoTokenizer.from_pretrained(config[esm2_select]['model_id'])
        self.max_len              = max_len
        self.mask_prob            = mask_prob
        self.start_from_N_terminal = start_from_N_terminal

        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls_token_id  = self.tokenizer.cls_token_id
        self.eos_token_id  = self.tokenizer.eos_token_id
        self.pad_token_id  = self.tokenizer.pad_token_id
        self.vocab_size    = self.tokenizer.vocab_size
        self.special_token_ids = {self.cls_token_id, self.eos_token_id, self.pad_token_id}

    def _open_hdf5(self):
        if self.h5py_files is None:
            self.h5py_files = [h5py.File(path, 'r') for path in self.h5py_paths]

    def _find_key_in_files(self, key):
        for f in self.h5py_files:
            if key in f:
                return f[key]
        raise KeyError(f"Key '{key}' not found in any HDF5 file")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        self._open_hdf5()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, list):
            return [self._process_single_item(i) for i in idx]
        return self._process_single_item(idx)

    def _process_single_item(self, idx):
        name = self.names[idx]
        seq  = self.seqs[idx]

        res_feat  = torch.tensor(self._find_key_in_files(f'{name}_res_feature')[:])
        pair_feat = torch.tensor(self._find_key_in_files(f'{name}_pair_feature')[:])

        if res_feat.shape[1] == 47:
            rmsf_feat  = res_feat[:, 2]
            valid_mask = rmsf_feat != -1
            if valid_mask.any():
                rmsf_feat[valid_mask] = rmsf_feat[valid_mask] / rmsf_feat[valid_mask].max()
                res_feat[:, 2] = rmsf_feat

        raw_input          = self.tokenizer(seq, return_tensors="pt")
        item               = {key: val.squeeze(0) for key, val in raw_input.items()}
        original_input_ids = item['input_ids'].clone()
        length             = original_input_ids.shape[0]

        if length > self.max_len:
            start = 0 if self.start_from_N_terminal else \
                torch.randint(0, length - self.max_len + 1, (1,)).item()
            original_input_ids    = original_input_ids[start:start + self.max_len]
            item['input_ids']     = item['input_ids'][start:start + self.max_len]
            item['attention_mask']= item['attention_mask'][start:start + self.max_len]
            res_feat  = res_feat[start:start + self.max_len, :]
            pair_feat = pair_feat[start:start + self.max_len, start:start + self.max_len, :]

        masked_input_ids, mask_positions = self._apply_masking(original_input_ids)

        return {
            'masked_input_ids':   masked_input_ids,
            'original_input_ids': original_input_ids,
            'attention_mask':     item['attention_mask'],
            'mask_positions':     mask_positions,
            'res_feat':           res_feat,
            'pair_feat':          pair_feat,
        }

    def _apply_masking(self, input_ids):
        masked_input_ids = input_ids.clone()
        seq_len          = len(input_ids)

        can_mask = torch.ones(seq_len, dtype=torch.bool)
        for i, token_id in enumerate(input_ids):
            if token_id.item() in self.special_token_ids:
                can_mask[i] = False

        mask_positions    = torch.zeros(seq_len, dtype=torch.bool)
        maskable_indices  = torch.where(can_mask)[0]

        if len(maskable_indices) > 0:
            num_to_mask  = max(1, int(len(maskable_indices) * self.mask_prob))
            mask_indices = maskable_indices[torch.randperm(len(maskable_indices))[:num_to_mask]]
            mask_positions[mask_indices] = True

            for idx in mask_indices:
                rand = random.random()
                if rand < 0.8:
                    masked_input_ids[idx] = self.mask_token_id
                elif rand < 0.9:
                    masked_input_ids[idx] = random.randint(4, self.vocab_size - 1)

        return masked_input_ids, mask_positions


DEFAULT_SOURCE_GROUPS = {
    'ATLAS_GPCRmd_PED_mdCATH': ['ATLAS', 'GPCRmd', 'PED', 'mdCATH'],
    'IDRome':                   ['IDRome'],
    'Proteinflow':              ['proteinflow_pdb', 'proteinflow_sabdab'],
}


def get_dataloader_cycle_iter(df, h5py_paths, esm2_select, max_len, batch_size,
                               n_update, device_id, mask_prob=0.15, num_workers=2,
                               source_groups=None):
    torch.manual_seed(n_update + device_id)
    torch.cuda.manual_seed(n_update + device_id)
    random.seed(n_update + device_id)

    if source_groups is None:
        source_groups = DEFAULT_SOURCE_GROUPS

    cycle_iters = {}
    for name, sources in source_groups.items():
        dataset = ProteinDatasetMLM(
            df[df['source'].isin(sources)].reset_index(drop=True),
            h5py_paths, esm2_select, max_len, mask_prob=mask_prob,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(dataset),
            collate_fn=collate_batch,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        cycle_iters[name] = iter(cycle(loader))

    return cycle_iters


def get_single_dataloader(df, h5py_paths, esm2_select, max_len, batch_size,
                          mask_prob=0.15, num_workers=2, shuffle=True):
    dataset = ProteinDatasetMLM(
        df, h5py_paths, esm2_select, max_len,
        mask_prob=mask_prob, start_from_N_terminal=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True,
    )
