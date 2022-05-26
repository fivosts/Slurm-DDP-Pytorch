"""Read all necessary SLURM environment variables to use in your app."""
import os
import ifcfg

try:
  MASTER_PORT         = int(os.environ.get("MASTER_PORT", 8738))
  MASTER_ADDR         = os.environ.get("MASTER_ADDR", "127.0.0.1")
  LOCAL_RANK          = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
  WORLD_RANK          = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
  WORLD_SIZE          = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
  if "GLOO_SOCKET_IFNAME" not in os.environ: # Used for DDP with CPUs only (use only for debugging)
    os.environ["GLOO_SOCKET_IFNAME"] = ifcfg.default_interface()['device']
  if "NCCL_SOCKET_IFNAME" not in os.environ: # DDP backend for CUDA. This is what you need for training.
    os.environ["NCCL_SOCKET_IFNAME"] = ifcfg.default_interface()['device']
except Exception as e:
  raise e
