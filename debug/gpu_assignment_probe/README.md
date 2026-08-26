# GPU assignment probe

This directory contains a lightweight probe for checking which worker and GPU a
`mytask` job actually uses.

## Direct run

```bash
cd /mlx_devbox/users/fengyuebo/playground/TIGER/debug/gpu_assignment_probe
bash run_probe.sh 0
bash run_probe.sh 1
```

The first argument is written to `CUDA_VISIBLE_DEVICES`. If it is omitted, the
job inherits the current environment.

## Submit through mytask

```bash
cd /mlx_devbox/users/fengyuebo/playground/TIGER/debug/gpu_assignment_probe
/mlx_devbox/users/fengyuebo/playground/MyTask/mytask run_probe.sh 0 -m "probe gpu 0"
/mlx_devbox/users/fengyuebo/playground/MyTask/mytask run_probe.sh 1 -m "probe gpu 1"
```

Then inspect the task id printed by `mytask`:

```bash
/mlx_devbox/users/fengyuebo/playground/MyTask/mytask show -w
/mlx_devbox/users/fengyuebo/playground/MyTask/mytask log <task_id>
```

## What to check

- `hostname`: the worker where the task process actually ran.
- `CUDA_VISIBLE_DEVICES`: the GPU id requested by `run_probe.sh`.
- `nvidia-smi`: all GPUs visible to the process environment.
- `torch.cuda.device_count`: the number of GPUs PyTorch can see after
  `CUDA_VISIBLE_DEVICES` filtering.
- `visible_device[0]`: when `CUDA_VISIBLE_DEVICES=1`, PyTorch usually remaps
  physical GPU 1 to logical `cuda:0`.

If `hostname` is not a GPU worker, `mytask` is not moving the job to a GPU
worker. If `CUDA_VISIBLE_DEVICES=1` but PyTorch still sees the wrong device,
the launch environment or container runtime is overriding GPU visibility.
