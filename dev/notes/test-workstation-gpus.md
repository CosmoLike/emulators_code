---
name: test-workstation-gpus
description: "The user's TEST workstation for the emulator drivers (a THIRD machine, distinct from the Mac dev box [[dev-machine-mac-m2-32gb]] and the NVWULF 8xH200 node in [[multi-gpu-sweep-pattern]]): 2x NVIDIA RTX 3060, 12 GB each, driver 535 / CUDA 12.2. GPU 0 (PCI 09:00.0) is an eGPU, SHARED (another user 'evan' runs a cocoa python there); GPU 1 (PCI A1:00.0) is the internal/DISPLAY GPU (Xorg + gnome-shell) the user wants their jobs to DEFAULT to. They asked 2026-07-01 how to make GPU 1 the default / avoid the eGPU while testing all the drivers. ANSWER: set CUDA_DEVICE_ORDER=PCI_BUS_ID (so CUDA's index matches nvidia-smi's PCI order -- CUDA's default 'fastest first' enumeration can differ, so '1' may not be the GPU you think) + CUDA_VISIBLE_DEVICES=1 (only GPU1, becomes cuda:0; sweep/bakeoff then see device_count==1 -> SERIAL path) OR =1,0 (both, GPU1 first as cuda:0, eGPU as cuda:1 -> the multi-GPU dispatch actually runs). The first-listed device = cuda:0 = default; our drivers respect the remap (pick_device / device_count / set_device). Must be set BEFORE the process starts (read at CUDA init), not in Python after import torch."
metadata:
  node_type: memory
  type: reference
---

The user's NVIDIA TEST workstation (where they test the emulator drivers before
NVWULF): 2x RTX 3060 (12 GB), driver 535.183.01, CUDA 12.2.

- GPU 0 = PCI `09:00.0` = an **eGPU**, shared (another user, `evan`, runs a cocoa
  python there ~104 MiB).
- GPU 1 = PCI `A1:00.0` = the internal / **display** GPU (Xorg + gnome-shell,
  ~227 MiB) the user wants their jobs to default to.

They asked (2026-07-01) how to make GPU 1 the default and keep off the eGPU while
testing all the drivers. The lever is `CUDA_VISIBLE_DEVICES` (picks which physical
GPUs CUDA sees and the order; the first listed becomes `cuda:0` = default), with
two gotchas:

- **`CUDA_DEVICE_ORDER=PCI_BUS_ID` first.** nvidia-smi numbers by PCI bus order,
  but CUDA's default enumeration ("fastest first") can differ, so CUDA's "1" may
  not be nvidia-smi's GPU 1. Pin it so the indices match.
- It is read at CUDA init, so set it on the command line / shell rc, not inside
  Python after `import torch`.

Recipes:
- `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python driver/...` — only
  GPU 1 (it becomes `cuda:0`). The sweep / bake-off see `device_count()==1` and
  take the SERIAL path (good for a clean correctness pass; does not exercise the
  multi-process fan-out).
- `... CUDA_VISIBLE_DEVICES=1,0 ...` — both GPUs, GPU 1 as `cuda:0`, the eGPU as
  `cuda:1`; `device_count()==2`, so the multi-GPU LPT / activation-split dispatch
  actually runs. Uses the shared eGPU lightly.

Our drivers need no change: `pick_device` / `torch.cuda.device_count()` /
`torch.cuda.set_device(k)` all operate on the remapped indices. Compute on the
display GPU (GPU 1) can make the desktop stutter under load (cosmetic).

**Why:** the user's actual test rig and the device-pinning recipe, so the next
session helps them run / debug the drivers on the right GPU without rediscovering
the `CUDA_DEVICE_ORDER` + `CUDA_VISIBLE_DEVICES` trick. Suggested test order: a
single-GPU correctness pass (`=1`), then one `=1,0` run to exercise the
multi-GPU dispatch end to end.
