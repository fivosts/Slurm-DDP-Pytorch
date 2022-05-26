"""
Example of initializing setting up a DDP model and training.
"""
import environment
import pytorch

if __name__ == "__main__":
  print("This function doesn't work, it only shows a simplified skeleton of DDP training in pytorch.")
  model = my_torch_model(config) # This should be a torch.nn.Module.
  if pytorch.num_nodes > 1:
    model = torch.nn.parallel.DistributedDataParallel(
      m,
      device_ids    = [pytorch.device],
      output_device = pytorch.device,
      find_unused_parameters = True, # Normally this flag shouldn't be used. Google to see what this does.
    )

    opt, lr_scheduler = get_optimizer_and_scheduler

    dataset = torch.Dataset(your_data)
    if environment.WORLD_SIZE == 1:
      sampler = torch.utils.data.RandomSampler(dataset, replacement = False)
    else:
      sampler = torch.utils.data.DistributedSampler(dataset)

    dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      batch_size = batch_size,
      sampler    = sampler,
    )

    model.train()
    for epoch in range(num_epochs):

      # In distributed mode, calling the set_epoch() method at
      # the beginning of each epoch before creating the DataLoader iterator
      # is necessary to make shuffling work properly across multiple epochs.
      # Otherwise, the same ordering will be always used.
      if pytorch.num_nodes > 1:
        loader.sampler.set_epoch(epoch)

      for batch in dataloder:

        # Move all inputs to torch device.
        inputs     = to_device(batch)
        # Run model step on batch
        step_out   = model(inputs)
        # Collect losses and backpropagate
        total_loss = step_out['total_loss'].mean()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters())
        opt.step()
        lr_scheduler.step()

        ## Collect tensors for logging.
        if pytorch.num_nodes > 1:
          torch.distributed.barrier()
          total_loss = [torch.zeros(tuple(step_out['total_loss'].shape), dtype = torch.float32).to(pytorch.device) for _ in range(torch.distributed.get_world_size())]
          torch.distributed.all_gather(total_loss, step_out['total_loss'])
        else:
          total_loss = step_out['total_loss'].unsqueeze(0).cpu()
        model.zero_grad()

      # End of Epoch
      if environment.WORLD_RANK == 0:
        saveCheckpoint(model, opt, lr_scheduler)


      if pytorch.num_nodes > 1:
        dataloader.sampler.set_epoch(epoch)
