import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--score_path", type=str, required=True)
args = parser.parse_args()

configs = [10, 20, 30, 40, 50, 60]

for cfg in configs:

    print(f"\nStarting config {cfg}\n")

    subprocess.run([
        "python3",
        "main.py",
        "--config",
        str(cfg),
        "--score_path",
        args.score_path
    ])

print("\nAll experiments completed!")