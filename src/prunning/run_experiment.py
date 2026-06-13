import subprocess
configs = [10,20,30,40,50, 60]

for cfg in configs:

    print(f"\nStarting config {cfg}\n")

    subprocess.run([
        "python3",
        "main.py",
        "--config",
        str(cfg)
    ])

print("\nAll experiments completed!")