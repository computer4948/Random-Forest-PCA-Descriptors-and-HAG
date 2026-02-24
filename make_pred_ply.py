import numpy as np
from ply import read_ply, write_ply
import os

# adapte si besoin
test_ply_path = os.path.join("..", "data", "MiniChallenge", "test", "MiniDijon9.ply")
pred_path = "MiniDijon9.txt"

ply = read_ply(test_ply_path)
pts = np.vstack((ply["x"], ply["y"], ply["z"])).T.astype(np.float32)
pred = np.loadtxt(pred_path, dtype=np.int32)

assert pts.shape[0] == pred.shape[0], f"Mismatch: {pts.shape[0]} points vs {pred.shape[0]} labels"

write_ply("MiniDijon9_pred.ply",
          (pts, pred),
          ["x", "y", "z", "pred"])

print("OK -> MiniDijon9_pred.ply")