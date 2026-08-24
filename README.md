# Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy

Build a complete training stack in pure NumPy that mirrors how modern frameworks scale models and fit them into limited memory. Implement an MLP with manual autograd, then add gradient accumulation, activation checkpointing, mixed precision, data-parallel all-reduce, and ZeRO-style optimizer sharding under realistic memory budgets.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_synthetic_regression_batch
- [x] **2.** init_mlp_params
- [x] **3.** linear_forward
- [x] **4.** relu_forward
- [x] **5.** mlp_forward
- [x] **6.** mse_loss_and_grad
- [x] **7.** linear_backward
- [x] **8.** relu_backward
- [x] **9.** first_linear_backward
- [x] **10.** mlp_backward
- [x] **11.** split_into_micro_batches
- [x] **12.** accumulate_gradients
- [x] **13.** scale_accumulated_gradients
- [x] **14.** grad_accumulation_step
- [x] **15.** mlp_forward_checkpointed
- [x] **16.** recompute_block_activations
- [x] **17.** mlp_backward_checkpointed
- [x] **18.** estimate_checkpointing_memory_savings
- [x] **19.** cast_to_half_precision
- [x] **20.** make_master_params
- [x] **21.** scale_loss
- [x] **22.** unscale_gradients
- [x] **23.** has_non_finite_gradients
- [x] **24.** mixed_precision_step
- [x] **25.** shard_dataset_across_workers
- [x] **26.** compute_local_gradients
- [x] **27.** all_reduce_mean
- [x] **28.** ring_all_reduce_mean
- [x] **29.** data_parallel_train_step
- [x] **30.** bucket_gradients
- [x] **31.** init_adam_state
- [x] **32.** partition_optimizer_state
- [x] **33.** local_shard_adam_update
- [x] **34.** all_gather_param_shards
- [x] **35.** zero_optimizer_step
- [x] **36.** compute_param_memory_bytes
- [x] **37.** compute_optimizer_memory_bytes
- [x] **38.** compute_peak_activation_memory_bytes
- [x] **39.** compare_memory_with_and_without_optimizations
- [x] **40.** full_distributed_training_loop

## Results

```
Data: x(32, 8), y(32, 4)
Params: W1(8, 16), b1(16,), W2(16, 4), b2(4,)
Initial MSE loss: 11.775703
Grad norms: W1=2.7308, b1=0.6080, W2=3.8400, b2=0.9442
Accumulated grad norm (W1): 2.7308
Checkpoint matches full backward: True
Checkpointing saves ~{'full_bytes': 5120, 'checkpoint_bytes': 1024, 'saved_bytes': 4096} bytes of activations
Loss after one data-parallel SGD step: 11.543701
ZeRO updated W2 norm: 2.7310
Memory comparison (bytes):
  baseline_bytes: 8176
  optimized_bytes: 1784
  breakdown_baseline: {'params': 848, 'optimizer': 1696, 'activations': 5632}
  breakdown_optimized: {'params': 424, 'optimizer': 848, 'activations': 512}
  savings_ratio: 0.7818003913894325
Loss history (10 steps): start=7.6543, end=7.2717
```
