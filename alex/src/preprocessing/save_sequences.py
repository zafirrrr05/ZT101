import os
import numpy as np

# each sequence 
def save_sequences(sequences, out_dir, start_index=0):

    os.makedirs(out_dir, exist_ok=True)

    saved_files = []

    for i, seq in enumerate(sequences):

        idx = start_index + i

        fname = f"seq_{idx:07d}.npz"
        path = os.path.join(out_dir, fname)

        np.savez_compressed(
            path,
            players_team1=seq["players_team1"],  # (T, 11, 4)
            players_team2=seq["players_team2"],  # (T, 11, 4)
            ball=seq["ball"],                    # (T, 4)
            referee=seq["referee"],              # (T, 4)
            start_frame=seq["start_frame"]       # added for debugging, can be removed later, used in testing_npz   
        )

        print("[SAVED]", path)
        saved_files.append(path)

    return saved_files