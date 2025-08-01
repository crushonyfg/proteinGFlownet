import argparse
import os
from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

import esm
import esm.inverse_folding
from datasets import load_dataset


class MegaScaleInverseFoldingDataset(Dataset):
    def __init__(self, hf_split, pdb_dir, max_items=None):
        """
        hf_split: HuggingFace Dataset split (e.g., dataset3_single['train'])
        pdb_dir: directory containing PDB files named exactly as WT_name entries
        max_items: optional cap for debugging
        """
        self.hf_split = hf_split
        self.pdb_dir = Path(pdb_dir)
        self.max_items = max_items
        self._cache = {}  # ??:WT_name -> (coords_tensor, wt_seq_str)

    def __len__(self):
        if self.max_items is not None:
            return min(len(self.hf_split), self.max_items)
        return len(self.hf_split)

    def _load_structure(self, wt_name):
        if wt_name in self._cache:
            return self._cache[wt_name]
        pdb_path = self.pdb_dir / wt_name
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB ?? '{wt_name}' ? {pdb_path} ????")
        coords_np, wt_seq = esm.inverse_folding.util.load_coords(str(pdb_path), None)
        # ????? torch.Tensor(CPU ?)
        coords_tensor = torch.tensor(coords_np, dtype=torch.float32)  # (L,3,3)
        self._cache[wt_name] = (coords_tensor, wt_seq)
        return coords_tensor, wt_seq

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        wt_name = item['WT_name']
        aa_seq = item['aa_seq']  # ????
        coords, wt_seq = self._load_structure(wt_name)
        return {
            'coords': coords,           # torch.Tensor (L,3,3)
            'target_seq': aa_seq,       # ???????
            'wt_seq': wt_seq,           # wild-type ?????(? PDB ??)
            'WT_name': wt_name,
            'meta': item
        }


def collate_fn(batch):
    # ????????,???? list,?? element ? dict
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="? ESM-IF1 ? MegaScale dataset3_single ????????")
    parser.add_argument('--pdb_dir', type=str, default='data/MegaScale/AlphaFold_model_PDBs',
                        help='?? WT PDB ?????(???????? WT_name)')
    parser.add_argument('--hf_dataset_name', type=str, default="dataset3_single",
                        help='MegaScale ?????,?? dataset3_single')
    parser.add_argument('--hf_cache_dir', type=str, default=None,
                        help='HuggingFace ????(??)')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='?????? checkpoint(.pt)')
    parser.add_argument('--output_path', type=str, default='checkpoints/esm_if1_finetuned.pt')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_every', type=int, default=1,
                        help='? N ? epoch ????')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda ? cpu')
    parser.add_argument('--max_train', type=int, default=None,
                        help='?????????(???)')
    parser.add_argument('--max_val', type=int, default=None,
                        help='?????????(???)')
    return parser.parse_args()


def evaluate(model, alphabet, dataloader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            for sample in batch:
                coords = sample['coords'].to(device)  # ??? tensor ?
                target_seq = sample['target_seq']
                loss_tensor, _ = esm.inverse_folding.util.get_sequence_loss(model, alphabet, coords, target_seq)
                loss = loss_tensor.mean()
                total_loss += loss.item()
                count += 1
    return total_loss / count if count > 0 else float('nan')
    
import torch.nn.functional as F
from esm.inverse_folding.util import CoordBatchConverter 

def get_sequence_loss_tensor(model, alphabet, coords, seq):
    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
    loss = F.cross_entropy(logits, target, reduction='none')
    return loss[0], target_padding_mask[0]



def train():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ?? MegaScale dataset3_single
    ds = load_dataset("RosettaCommons/MegaScale",
                      name=args.hf_dataset_name,
                      data_dir=args.hf_dataset_name,
                      cache_dir=args.hf_cache_dir)
    train_split = ds['train']
    val_split = ds['val']

    # ?? Dataset
    train_ds = MegaScaleInverseFoldingDataset(train_split, args.pdb_dir, max_items=args.max_train)
    val_ds = MegaScaleInverseFoldingDataset(val_split, args.pdb_dir, max_items=args.max_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)

    # ????? alphabet
    model, alphabet = esm.pretrained.load_model_and_alphabet_local('weights/esm_if1_gvp4_t16_142M_UR50.pt')
    model.train()
    model = model.to(device)

    # ????? checkpoint ???
    if args.weights_path:
        ckpt = torch.load(args.weights_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)  # ???? mismatch

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # ????
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        examples = 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] training", leave=True)
        for batch in pbar:
            optimizer.zero_grad()
            batch_loss = 0.0
            for sample in batch:
                coords = sample['coords'].to(device)  # tensor -> device
                target_seq = sample['target_seq']
                # loss_tensor, _ = esm.inverse_folding.util.get_sequence_loss(model, alphabet, coords, target_seq)
                loss_tensor, _ = get_sequence_loss_tensor(model, alphabet, coords, target_seq)
                loss = loss_tensor.mean()
                batch_loss += loss
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += batch_loss.item()
            examples += 1
            pbar.set_postfix({'batch_loss': batch_loss.item(), 'avg_loss': running_loss / examples})

        avg_train_loss = running_loss / examples if examples > 0 else float('nan')
        print(f"Epoch {epoch} train loss: {avg_train_loss:.4f}")

        # ??
        val_loss = evaluate(model, alphabet, val_loader, device)
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")

        # ??
        if epoch % args.save_every == 0:
            out_path = Path(args.output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(out_path))
            print(f"Saved checkpoint to {out_path}")

    # ????
    torch.save(model.state_dict(), args.output_path)
    print(f"Final model saved to {args.output_path}")


if __name__ == '__main__':
    train()
