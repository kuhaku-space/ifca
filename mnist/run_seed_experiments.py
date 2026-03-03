import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run MNIST IFCA experiments with multiple seeds"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="List of seeds to run (default: 0 1 2 3 4)",
    )
    args = parser.parse_args()

    base_output_dir = "output"

    for seed in args.seeds:
        print(f"\n{'=' * 50}")
        print(f"🚀 Starting MNIST experiment with seed: {seed}")
        print(f"{'=' * 50}")

        # Create a specific output directory for this seed
        project_dir = os.path.join(base_output_dir, f"seed_{seed}")
        os.makedirs(project_dir, exist_ok=True)

        # Command to run the IFCA clustering for MNIST
        # Based on GEMINI.md, use `uv run`
        cmd = [
            "uv",
            "run",
            "train_cluster_mnist.py",
            "--data-seed",
            str(seed),
            "--train-seed",
            str(seed),
            "--project-dir",
            project_dir,
        ]

        print(f"Running: {' '.join(cmd)}\n")

        log_file_path = os.path.join(project_dir, "stdout.log")
        print(f"Logging output to: {log_file_path}")

        try:
            # Run the training script, redirecting stdout/stderr to a file
            with open(log_file_path, "w") as log_file:
                subprocess.run(
                    cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT
                )
            print(f"\n✅ Finished seed {seed} successfully.")
            print(f"Results saved in {project_dir}/\n")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error occurred while running seed {seed}: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Experiment interrupted by user during seed {seed}.")
            sys.exit(1)

    print(f"{'=' * 50}")
    print("🎉 All seed experiments completed!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
