import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.pass_utils import detect_pass_events, extract_pass_window, pass_success


class PassDataset(Dataset):

    def __init__(self, root):

        self.samples = []

        files = [os.path.join(root,f) for f in os.listdir(root) if f.endswith(".npz")]

        for f in files:
            d = np.load(f)

            players = torch.from_numpy(d["players"])
            ball    = torch.from_numpy(d["ball"])

            events = detect_pass_events(players, ball)

            for t in events:
                w = extract_pass_window(players, ball, t)
                if w is None:
                    continue

                y = pass_success(players, ball, t)
                if y is None:
                    continue

                self.samples.append({
                    "players": w["players"],
                    "ball": w["ball"],
                    "label": y
                })

        print("PassDataset size:", len(self.samples))        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]

        return {
            "players": s["players"].float(),
            "ball": s["ball"].float(),
            "label": torch.tensor(s["label"], dtype=torch.long)
        }