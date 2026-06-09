import subprocess
configs = [50, 60]

for cfg in configs:

    print(f"\nStarting config {cfg}\n")

    subprocess.run([
        "python3",
        "main.py",
        "--config",
        str(cfg)
    ])

print("\nAll experiments completed!")