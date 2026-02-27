import torch

def build_space_targets(players, ball, grid_xy):
    """
    players: [T,11,4]   (already normalized to meters later)
    ball:    [T,4]
    grid_xy: [H,W,2]
    """

    T = players.shape[0]
    H, W = grid_xy.shape[:2]

    targets = torch.zeros(T, H, W)

    for t in range(T):

        p = players[t][:, :2]   # [11,2]
        b = ball[t][:2]

        diff = grid_xy.view(-1,2)[:,None,:] - p[None,:,:]
        d_def = torch.norm(diff, dim=-1).min(dim=1)[0]
        d_def = d_def.view(H,W)

        d_ball = torch.norm(grid_xy - b, dim=-1)

        targets[t] = torch.sigmoid(
            1.5 * d_def - 0.8 * d_ball
        )

    return targets