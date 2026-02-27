import torch
from torch.utils.data import DataLoader

from src.models.team_tactical_net import TacticalModel
from src.training.pass_dataset import PassDataset


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PassDataset("data/sequences")
    
    if len(ds) == 0:
        print("No pass quality samples found. Did you generate labels?")
        return

    loader = DataLoader(ds, batch_size=8, shuffle=True)

    model = TacticalModel(dim=64).to(device)

    ckpt = torch.load("checkpoints/ssl_backbone.pt", map_location=device)
    model.backbone.load_state_dict(ckpt)

    for p in model.backbone.parameters():
        p.requires_grad = False

    optim = torch.optim.Adam(model.pass_head.parameters(), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss()

    for epoch in range(10):

        for batch in loader:

            players = batch["players"].to(device)
            ball = batch["ball"].to(device)
            y = batch["label"].to(device)

            tokens = model.backbone(players, ball)

            center = tokens[:, tokens.shape[1]//2]

            logits = model.pass_head(center)

            loss = ce(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        print("epoch", epoch, "loss", loss.item())


if __name__ == "__main__":
    main()