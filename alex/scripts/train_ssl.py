import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.ssl_dataset import TeamSequenceDataset
from src.training.ssl_collate import SSLTaskSampler
from src.training.ssl_trainer import SSLTrainer


def main():

    ds = TeamSequenceDataset("data/sequences", min_len=40)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        collate_fn=SSLTaskSampler(),
        num_workers=0,   # <- important on Windows
        pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = SSLTrainer(device=device)
    pbar = tqdm(enumerate(loader), total=501)

    for step, batch in pbar:
        loss = trainer.train_step(batch)
        pbar.set_description(
            f"step {step} | task {batch['task']} | loss {loss:.4f}"
        )
        if step >= 200:
            break


if __name__ == "__main__":
    main()