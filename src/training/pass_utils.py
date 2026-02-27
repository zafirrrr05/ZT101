import torch

def possessor(players_t, ball_t):
    # players_t: [11,4]
    # ball_t: [4]

    d = torch.norm(players_t[:, :2] - ball_t[:2], dim=1)
    return torch.argmin(d).item()

def detect_pass_events(players, ball,
                        min_ball_speed=0.02,
                        min_gap=5):
    """
    players: [T,11,4]
    ball:    [T,4]
    """

    events = []

    last_event_t = -999

    prev_owner = possessor(players[0], ball[0])

    for t in range(1, players.shape[0]):

        owner = possessor(players[t], ball[t])

        speed = torch.norm(ball[t,2:4]).item()

        if owner != prev_owner and speed > min_ball_speed:

            if t - last_event_t >= min_gap:
                events.append(t)
                last_event_t = t

        prev_owner = owner

    return events

def extract_pass_window(players, ball, t, pre=10, post=20):

    start = t - pre
    end   = t + post

    if start < 0 or end >= players.shape[0]:
        return None

    return {
        "players": players[start:end],
        "ball": ball[start:end],
        "t_rel": pre
    }

def pass_success(players, ball, t_pass,
                 horizon=15):

    T = players.shape[0]

    if t_pass + horizon >= T:
        return None

    owner_now = possessor(players[t_pass], ball[t_pass])
    owner_later = possessor(players[t_pass + horizon], ball[t_pass + horizon])

    return int(owner_now == owner_later)

