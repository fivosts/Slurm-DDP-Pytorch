"""Example of how DDP should be initialized in Pytorch."""
import torch
import environment

device        = None
num_gpus      = None
num_nodes     = None

def initPytorch() -> None:
  global device
  global num_gpus
  global num_nodes

  tcp_store = torch.distributed.TCPStore(
    environment.MASTER_ADDR,
    environment.MASTER_PORT,
    environment.WORLD_SIZE,
    environment.WORLD_RANK == 0
  )
  torch.distributed.init_process_group(
    backend    = "nccl", # This is the default backend for DDP-GPU training. For DDP-CPU training only (suggested for faster debugging, use "gloo")
    store      = tcp_store,
    rank       = environment.WORLD_RANK,
    world_size = environment.WORLD_SIZE,
  )
  num_nodes = torch.distributed.get_world_size()
  num_gpus  = torch.cuda.device_count()

  if num_gpus == 0:
    device = torch.device('cpu', environment.LOCAL_RANK)
  else:
    device = torch.device("cuda", environment.LOCAL_RANK)
    torch.cuda.set_device(device)
  return
