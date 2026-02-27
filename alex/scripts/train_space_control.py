import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.team_tactical_net import TacticalModel
from src.training.ssl_dataset import TeamSequenceDataset


# -------------------------
# grid builder (meters later, pixels for now)
# -------------------------
def build_grid(H, W, device):
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)  # [H,W,2]
    return grid


# -------------------------
# pseudo target generator
# -------------------------
def build_space_targets(players, ball, grid):
    """
    players : [B,T,11,4]
    ball    : [B,T,4]
    grid    : [H,W,2]
    """

    B, T, N, _ = players.shape
    H, W, _ = grid.shape

    targets = torch.zeros(B, T, H, W, device=players.device)

    grid_flat = grid.view(-1, 2)  # [HW,2]

    for b in range(B):
        for t in range(T):

            p = players[b, t, :, :2]     # [11,2]
            bpos = ball[b, t, :2]        # [2]

            diff = grid_flat[:, None, :] - p[None, :, :]
            d_def = torch.norm(diff, dim=-1).min(dim=1)[0]  # [HW]
            d_def = d_def.view(H, W)

            d_ball = torch.norm(grid - bpos, dim=-1)

            targets[b, t] = torch.sigmoid(
                1.5 * d_def - 0.8 * d_ball
            )

    return targets


# -------------------------
# main training
# -------------------------
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # same dataset style as SSL
    ds = TeamSequenceDataset("data/sequences", min_len=40)

    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    model = TacticalModel(dim=64).to(device)

    # load SSL backbone
    ckpt = torch.load("checkpoints/ssl_backbone.pt", map_location=device)
    model.backbone.load_state_dict(ckpt)

    # freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    model.space_head.train()

    optimizer = torch.optim.Adam(
        model.space_head.parameters(),
        lr=1e-3
    )

    H, W = 24, 16
    grid = build_grid(H, W, device)

    for epoch in range(10):

        for i, batch in enumerate(loader):

            players = batch["players"].to(device)  # [B,T,11,4]
            ball = batch["ball"].to(device)        # [B,T,4]

            with torch.no_grad():
                targets = build_space_targets(players, ball, grid)

            tokens = model.backbone(players, ball)      # [B,T,D]
            pred = model.space_head(tokens)             # [B,T,H,W]

            loss = F.mse_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    f"epoch {epoch} step {i} "
                    f"space_loss {loss.item():.4f}"
                )

    torch.save(
        model.space_head.state_dict(),
        "checkpoints/space_head.pt"
    )


if __name__ == "__main__":
    main()