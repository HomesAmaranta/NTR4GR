#!/usr/bin/env python3
import argparse
import os
import platform
import socket
import subprocess
import sys
import time


def run_cmd(cmd):
    try:
        out = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return f"<failed to run {' '.join(cmd)}: {exc}>"
    return out.stdout.strip() or "<no output>"


def print_section(title):
    print(f"\n===== {title} =====", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Probe which worker/GPU a mytask-launched job can see."
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=0,
        help="Seconds to keep the process alive after probing.",
    )
    args = parser.parse_args()

    print_section("process")
    print(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"hostname: {socket.gethostname()}", flush=True)
    print(f"fqdn: {socket.getfqdn()}", flush=True)
    print(f"pid: {os.getpid()}", flush=True)
    print(f"cwd: {os.getcwd()}", flush=True)
    print(f"python: {sys.executable}", flush=True)
    print(f"platform: {platform.platform()}", flush=True)

    print_section("environment")
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
    ):
        print(f"{key}: {os.environ.get(key, '<unset>')}", flush=True)

    print_section("nvidia-smi")
    print(
        run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        flush=True,
    )

    print_section("torch")
    try:
        import torch

        print(f"torch: {torch.__version__}", flush=True)
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}", flush=True)
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}", flush=True)
        if torch.cuda.is_available():
            current = torch.cuda.current_device()
            print(f"torch.cuda.current_device: {current}", flush=True)
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                print(
                    f"visible_device[{idx}]: name={props.name}, "
                    f"total_memory={props.total_memory}",
                    flush=True,
                )
            x = torch.empty((1,), device="cuda")
            print(f"allocated_tensor_device: {x.device}", flush=True)
    except Exception as exc:
        print(f"torch_probe_error: {type(exc).__name__}: {exc}", flush=True)

    if args.sleep > 0:
        print_section("sleep")
        print(f"sleeping {args.sleep}s for external inspection", flush=True)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
