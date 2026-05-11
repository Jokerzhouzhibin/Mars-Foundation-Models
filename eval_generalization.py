import os
import subprocess
import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor


def run_gpu_tasks(gpu_id, task_list, args):
    """
    Run assigned tasks sequentially on a specific GPU.
    """
    print(f"🔧 [GPU {gpu_id}] Assigned {len(task_list)} tasks queue.")
    
    for task in task_list:
        dataset, model, fraction, lr, strategy = task
        print(f"▶️  [GPU {gpu_id}] Start: {model} ({strategy}) on {dataset} (Frac={fraction}, LR={lr})")
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        cmd = [
            "python", "train_net.py",
            "--dataset", dataset,
            "--model", model,
            "--mode", args.mode,
            "--lr", str(lr),
            "--strategy", strategy,
            "--data_fraction", fraction,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--patience", str(args.patience)
        ]
        
        # Suppress outputs similar to full-train.sh > /dev/null 2>&1
        try:
            subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ [GPU {gpu_id}] DONE: {model} ({strategy}) on {dataset} (Frac={fraction}, LR={lr})")
        except subprocess.CalledProcessError:
            print(f"❌ [GPU {gpu_id}] FAILED: {model} ({strategy}) on {dataset} (Frac={fraction}, LR={lr})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Model Generalization with different dataset sizes, LRs, and strategies.")
    parser.add_argument("--mode", type=str, default="linear", choices=["linear", "full"], help="Training mode")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per run")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--gpus", type=str, default="1,2,3", help="Comma-separated list of GPUs to use (e.g., 0,1,2,3)")
    
    args = parser.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    num_gpus = len(gpus)
    
    if num_gpus == 0:
        print("❌ No GPUs provided.")
        return

    # 1. Parameter Grid
    datasets = [
        "mb-atmospheric_dust_cls_rdr",
        "mb-change_cls_ctx",
        "mb-change_cls_hirise",
        "mb-domars16k",
        "mb-frost_cls",
        "mb-landmark_cls"
    ]
    models = (
        "vit_l",
        "swin_v2",
        "satmae",
        "mars_mae",
        "siglip2",
        "openclip",
        "dinov1_b8",
        "dinov2_l14",
        "dinov3_lvd",
        "dinov3_sat",
        "dinov3_lvd_1cha",
        "dinov3_sat_1cha",
    )
    data_fractions = ["0.01x", "0.05x", "0.20x", "0.50x"]
    learning_rates = [1e-3, 1e-4, 1e-5]
    strategies = ["concat_3"]

    # 2. Generate all unique tasks
    tasks = list(itertools.product(datasets, models, data_fractions, learning_rates, strategies))

    print(f"🚀 Starting Parallel Evaluation on GPUs {gpus}...")
    print(f"🔥 Total Tasks: {len(tasks)} | Workers: {num_gpus}")

    # 3. Task Distribution (Round Robin)
    gpu_queues = {gpu: [] for gpu in gpus}
    for i, task in enumerate(tasks):
        gpu_id = gpus[i % num_gpus]
        gpu_queues[gpu_id].append(task)

    # 4. Parallel Execution with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id in gpus:
            if gpu_queues[gpu_id]:
                futures.append(executor.submit(run_gpu_tasks, gpu_id, gpu_queues[gpu_id], args))

        # Wait for all futures to complete
        for future in futures:
            future.result()

    print("\n🎉 All generalization evaluation tasks finished across designated GPUs!")


if __name__ == "__main__":
    main()
