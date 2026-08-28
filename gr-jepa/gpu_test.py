import os
import socket
import json
import torch

info = {
    "hostname": socket.gethostname(),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "gpus": [],
}

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_b, total_b = torch.cuda.mem_get_info(i)
        used_b = total_b - free_b

        info["gpus"].append({
            "index": i,
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_mib": round(total_b / 1024 / 1024),
            "free_mib": round(free_b / 1024 / 1024),
            "used_mib": round(used_b / 1024 / 1024),
            "free_percent": round(free_b * 100.0 / total_b, 2),
        })

print(json.dumps(info, indent=2, ensure_ascii=False))