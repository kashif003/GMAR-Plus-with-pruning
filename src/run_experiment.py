import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--score_path", type=str, required=True)
args = parser.parse_args()

configs = [60]

for cfg in configs:

    print(f"\nStarting config {cfg}\n")

    subprocess.run([
        "python3",
        "src/main.py",
        "--config",
        str(cfg),
        "--score_path",
        args.score_path
    ])

print("\nAll experiments completed!")