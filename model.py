"""
Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_synthetic_regression_batch
import numpy as np

def make_synthetic_regression_batch(batch_size, in_dim, out_dim, seed):
    """Return (x, y) where x is (batch_size, in_dim) and y is (batch_size, out_dim) float64."""
    np.random.seed(seed)

    x = np.random.randn(batch_size, in_dim).astype(np.float64)

    # Hidden linear teacher mapping.
    teacher_w = np.random.randn(in_dim, out_dim).astype(np.float64)

    # Small Gaussian observation noise.
    noise = 0.1 * np.random.randn(batch_size, out_dim)

    y = x @ teacher_w + noise

    return x.astype(np.float64), y.astype(np.float64)

# Step 2 - init_mlp_params
def init_mlp_params(in_dim, hidden_dim, out_dim, seed):
    """Return He-initialized parameters for a two-layer MLP."""
    np.random.seed(seed)

    # He initialization for the first layer.
    W1 = np.random.randn(in_dim, hidden_dim) * np.sqrt(2.0 / in_dim)

    # Zero bias for the first layer.
    b1 = np.zeros(hidden_dim, dtype=np.float64)

    # He initialization for the second layer.
    W2 = np.random.randn(hidden_dim, out_dim) * np.sqrt(2.0 / hidden_dim)

    # Zero bias for the second layer.
    b2 = np.zeros(out_dim, dtype=np.float64)

    return {
        "W1": W1.astype(np.float64),
        "b1": b1,
        "W2": W2.astype(np.float64),
        "b2": b2,
    }

# Step 3 - linear_forward
def linear_forward(x, w, b):
    """Apply a fully connected layer: y = x @ w + b."""
    return x @ w + b

# Step 4 - relu_forward
def relu_forward(x):
    """Apply ReLU elementwise."""
    return np.maximum(0.0, x)

# Step 5 - mlp_forward
def mlp_forward(x, params):
    """Run a two-layer MLP: Linear -> ReLU -> Linear."""
    z1 = linear_forward(x, params["W1"], params["b1"])
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, params["W2"], params["b2"])

    cache = {
        "x": x,
        "z1": z1,
        "a1": a1,
        "z2": z2,
    }

    return z2, cache

# Step 6 - mse_loss_and_grad
def mse_loss_and_grad(y_pred, y_true):
    """Compute mean squared error loss and its gradient with respect to y_pred."""
    diff = y_pred - y_true

    loss = float(np.mean(diff ** 2))

    dy_pred = 2.0 * diff / diff.size

    return loss, dy_pred

# Step 7 - linear_backward
def linear_backward(d_out, x, w):
    """Backpropagate through a linear layer: y = x @ w + b."""
    dx = d_out @ w.T
    dw = x.T @ d_out
    db = np.sum(d_out, axis=0)

    return dx, dw, db

# Step 8 - relu_backward
def relu_backward(d_out, z):
    """Backpropagate through ReLU using the pre-activation z."""
    return d_out * (z > 0)

# Step 9 - first_linear_backward
def first_linear_backward(d_z1, x, w1):
    """Backpropagate through the first linear layer."""
    dx = d_z1 @ w1.T
    dW1 = x.T @ d_z1
    db1 = np.sum(d_z1, axis=0)

    return dx, dW1, db1

# Step 10 - mlp_backward
def mlp_backward(dy_pred, cache, params):
    """Run the full backward pass of the two-layer MLP."""
    # Backpropagate through the second linear layer.
    da1, dW2, db2 = linear_backward(
        dy_pred,
        cache["a1"],
        params["W2"],
    )

    # Backpropagate through ReLU.
    dz1 = relu_backward(da1, cache["z1"])

    # Backpropagate through the first linear layer.
    dx, dW1, db1 = first_linear_backward(
        dz1,
        cache["x"],
        params["W1"],
    )

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }

# Step 11 - split_into_micro_batches
def split_into_micro_batches(x, y, micro_batch_size):
    """Split (x, y) into contiguous micro-batches."""
    micro_batches = []

    for start in range(0, len(x), micro_batch_size):
        end = start + micro_batch_size
        micro_batches.append((x[start:end], y[start:end]))

    return micro_batches

# Step 12 - accumulate_gradients
def accumulate_gradients(accum_grads, new_grads):
    """Accumulate gradients by elementwise summation."""
    if accum_grads is None:
        return {key: value.copy() for key, value in new_grads.items()}

    return {
        key: accum_grads[key] + new_grads[key]
        for key in new_grads
    }

# Step 13 - scale_accumulated_gradients
def scale_accumulated_gradients(accum_grads, num_micro_batches):
    """Average accumulated gradients over the number of micro-batches."""
    return {
        key: value / num_micro_batches
        for key, value in accum_grads.items()
    }

# Step 14 - grad_accumulation_step
def grad_accumulation_step(x, y, params, micro_batch_size):
    """Run one logical gradient step using micro-batch accumulation."""
    micro_batches = split_into_micro_batches(
        x,
        y,
        micro_batch_size,
    )

    accum_grads = None

    for x_mb, y_mb in micro_batches:
        y_pred, cache = mlp_forward(x_mb, params)

        _, dy_pred = mse_loss_and_grad(
            y_pred,
            y_mb,
        )

        new_grads = mlp_backward(
            dy_pred,
            cache,
            params,
        )

        accum_grads = accumulate_gradients(
            accum_grads,
            new_grads,
        )

    return scale_accumulated_gradients(
        accum_grads,
        len(micro_batches),
    )

# Step 15 - mlp_forward_checkpointed
def mlp_forward_checkpointed(x, params):
    """Run the MLP forward pass while caching only the block input x."""
    z1 = linear_forward(x, params["W1"], params["b1"])
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, params["W2"], params["b2"])

    cache = {
        "x": x,
    }

    return z2, cache

# Step 16 - recompute_block_activations
def recompute_block_activations(x, params):
    """Recompute the MLP block activations from the saved input x."""
    z1 = linear_forward(x, params["W1"], params["b1"])
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, params["W2"], params["b2"])

    return {
        "x": x,
        "z1": z1,
        "a1": a1,
        "z2": z2,
    }

# Step 17 - mlp_backward_checkpointed
def mlp_backward_checkpointed(dy_pred, light_cache, params):
    """Recompute activations and run the standard MLP backward pass."""
    cache = recompute_block_activations(
        light_cache["x"],
        params,
    )

    return mlp_backward(
        dy_pred,
        cache,
        params,
    )

# Step 18 - estimate_checkpointing_memory_savings
def estimate_checkpointing_memory_savings(batch_size, in_dim, hidden_dim, out_dim, dtype_bytes):
    """Estimate activation memory in bytes for full vs checkpointed forward."""
    full_bytes = int(batch_size * (in_dim + 2 * hidden_dim) * dtype_bytes)
    checkpoint_bytes = int(batch_size * in_dim * dtype_bytes)
    saved_bytes = full_bytes - checkpoint_bytes

    return {
        "full_bytes": full_bytes,
        "checkpoint_bytes": checkpoint_bytes,
        "saved_bytes": saved_bytes,
    }

# Step 19 - cast_to_half_precision
def cast_to_half_precision(values):
    """Return a new dict with all arrays converted to float16."""
    return {
        key: value.astype(np.float16)
        for key, value in values.items()
    }

# Step 20 - make_master_params
def make_master_params(params):
    """Return independent float32 master copies of all parameters."""
    return {
        key: value.astype(np.float32).copy()
        for key, value in params.items()
    }

# Step 21 - scale_loss
def scale_loss(loss, dy_pred, scale):
    """Scale the loss and upstream gradient by a fixed loss scale."""
    scaled_loss = loss * scale
    scaled_dy_pred = dy_pred * scale

    return scaled_loss, scaled_dy_pred

# Step 22 - unscale_gradients
def unscale_gradients(grads, scale):
    """Unscale gradients and return them as a new float32 dict."""
    return {
        key: (value / scale).astype(np.float32)
        for key, value in grads.items()
    }

# Step 23 - has_non_finite_gradients
def has_non_finite_gradients(grads):
    """Return True if any gradient contains NaN or Inf."""
    return any(not np.all(np.isfinite(grad)) for grad in grads.values())

# Step 24 - mixed_precision_step
def mixed_precision_step(x, y, master_params, scale, lr):
    """Run one mixed-precision training step with an fp32 master copy."""
    master_params = {
        key: value.astype(np.float32).copy()
        for key, value in master_params.items()
    }

    half_params = cast_to_half_precision(master_params)
    x_half = x.astype(np.float16)
    y_half = y.astype(np.float16)

    y_pred, cache = mlp_forward(x_half, half_params)

    loss, dy_pred = mse_loss_and_grad(y_pred, y_half)

    _, scaled_dy_pred = scale_loss(loss, dy_pred, scale)

    scaled_grads = mlp_backward(
        scaled_dy_pred,
        cache,
        half_params,
    )

    grads = unscale_gradients(scaled_grads, scale)

    if has_non_finite_gradients(grads):
        return float(loss), master_params, True

    new_master_params = {
        key: (master_params[key] - lr * grads[key]).astype(np.float32)
        for key in master_params
    }

    return float(loss), new_master_params, False

# Step 25 - shard_dataset_across_workers
def shard_dataset_across_workers(x, y, num_workers):
    """Split x and y into num_workers contiguous shards along axis 0."""
    n = len(x)
    base_size = n // num_workers
    remainder = n % num_workers

    shards = []
    start = 0

    for worker in range(num_workers):
        shard_size = base_size + (1 if worker < remainder else 0)
        end = start + shard_size

        shards.append((x[start:end], y[start:end]))
        start = end

    return shards

# Step 26 - compute_local_gradients
def compute_local_gradients(x, y, params):
    """Compute parameter gradients for one worker's data shard."""
    y_pred, cache = mlp_forward(x, params)
    _, dy_pred = mse_loss_and_grad(y_pred, y)
    grads = mlp_backward(dy_pred, cache, params)
    return grads

# Step 27 - all_reduce_mean
def all_reduce_mean(per_worker_grads):
    """Average gradient dictionaries elementwise across workers."""
    num_workers = len(per_worker_grads)

    return {
        key: sum(grads[key] for grads in per_worker_grads) / num_workers
        for key in per_worker_grads[0]
    }

# Step 28 - ring_all_reduce_mean
def ring_all_reduce_mean(per_worker_arrays):
    """Average arrays across workers using simulated ring reduce-scatter and all-gather."""
    num_workers = len(per_worker_arrays)

    if num_workers == 0:
        raise ValueError("per_worker_arrays must not be empty")

    shape = per_worker_arrays[0].shape

    if any(arr.shape != shape for arr in per_worker_arrays):
        raise ValueError("All worker arrays must have the same shape")

    if num_workers == 1:
        return per_worker_arrays[0].copy()

    flat_arrays = [arr.reshape(-1) for arr in per_worker_arrays]

    # Split into chunks as evenly as possible. np.array_split handles
    # cases where the number of elements is not divisible by workers.
    chunks = [
        np.array_split(flat_array, num_workers)
        for flat_array in flat_arrays
    ]

    # Ring reduce-scatter:
    # Worker i ultimately owns chunk i, containing the sum from every worker.
    reduced_chunks = []

    for chunk_id in range(num_workers):
        total = chunks[0][chunk_id].copy()

        for worker in range(1, num_workers):
            total = total + chunks[worker][chunk_id]

        reduced_chunks.append(total)

    # Ring all-gather:
    # Every worker would circulate the reduced chunks around the ring.
    # Since all workers must end with the same result, reconstruct the
    # gathered flat array from the reduced chunks.
    result = np.concatenate(reduced_chunks) / num_workers

    return result.reshape(shape)

# Step 29 - data_parallel_train_step
def data_parallel_train_step(x, y, params, num_workers, lr):
    """Perform one synchronous data-parallel SGD update."""
    shards = shard_dataset_across_workers(x, y, num_workers)

    per_worker_grads = [
        compute_local_gradients(x_shard, y_shard, params)
        for x_shard, y_shard in shards
    ]

    mean_grads = all_reduce_mean(per_worker_grads)

    new_params = {
        key: params[key] - lr * mean_grads[key]
        for key in params
    }

    return new_params

# Step 30 - bucket_gradients
def bucket_gradients(grads, bucket_size):
    """Pack flattened gradients into greedy communication buckets."""
    buckets = []
    meta = []

    current_values = []
    current_size = 0
    bucket_index = 0

    for name in sorted(grads.keys()):
        grad = grads[name]
        flat = grad.reshape(-1)
        size = flat.size

        # Start a new bucket if the current bucket cannot fit this tensor.
        if current_values and current_size + size > bucket_size:
            buckets.append(np.concatenate(current_values))
            bucket_index += 1
            current_values = []
            current_size = 0

        start = current_size
        end = start + size

        current_values.append(flat)
        current_size = end

        meta.append((name, grad.shape, start, end, bucket_index))

    # Flush the final bucket.
    if current_values:
        buckets.append(np.concatenate(current_values))

    return buckets, meta

# Step 31 - init_adam_state
def init_adam_state(params):
    """Build Adam optimizer state with zero first and second moments."""
    m = {key: np.zeros_like(value) for key, value in params.items()}
    v = {key: np.zeros_like(value) for key, value in params.items()}

    return {
        "m": m,
        "v": v,
        "t": 0,
    }

# Step 32 - partition_optimizer_state
def partition_optimizer_state(state, num_workers):
    """Partition Adam moment tensors into contiguous shards across workers."""
    workers = [
        {
            "m": {},
            "v": {},
            "shard_slices": {},
            "shapes": {},
            "t": state["t"],
        }
        for _ in range(num_workers)
    ]

    for name in state["m"]:
        m_flat = state["m"][name].reshape(-1)
        v_flat = state["v"][name].reshape(-1)
        size = m_flat.size

        base_size = size // num_workers
        remainder = size % num_workers

        start = 0

        for worker_idx in range(num_workers):
            shard_size = base_size + (1 if worker_idx < remainder else 0)
            end = start + shard_size

            workers[worker_idx]["m"][name] = m_flat[start:end].copy()
            workers[worker_idx]["v"][name] = v_flat[start:end].copy()
            workers[worker_idx]["shard_slices"][name] = (start, end)
            workers[worker_idx]["shapes"][name] = state["m"][name].shape

            start = end

    return workers

# Step 33 - local_shard_adam_update
def local_shard_adam_update(params, grads, worker_state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """Apply one Adam update to the local parameter shards."""
    t = worker_state["t"] + 1

    updated_param_shards = {}

    for name in worker_state["m"]:
        start, end = worker_state["shard_slices"][name]

        grad_flat = grads[name].reshape(-1)
        grad_shard = grad_flat[start:end]

        m = worker_state["m"][name]
        v = worker_state["v"][name]

        m *= beta1
        m += (1.0 - beta1) * grad_shard

        v *= beta2
        v += (1.0 - beta2) * (grad_shard ** 2)

        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        param_flat = params[name].reshape(-1)
        param_shard = param_flat[start:end]

        updated_param_shards[name] = (
            param_shard - lr * m_hat / (np.sqrt(v_hat) + eps)
        )

    updated_worker_state = {
        "m": {name: value.copy() for name, value in worker_state["m"].items()},
        "v": {name: value.copy() for name, value in worker_state["v"].items()},
        "t": t,
        "shard_slices": worker_state["shard_slices"].copy(),
        "shapes": worker_state["shapes"].copy(),
    }

    return updated_param_shards, updated_worker_state

# Step 34 - all_gather_param_shards
def all_gather_param_shards(param_shards_per_worker, shapes, shard_slices_per_worker):
    """Reassemble full parameter tensors from per-worker 1D shards."""
    params = {}

    for name, shape in shapes.items():
        total_size = int(np.prod(shape))
        full_flat = np.empty(total_size, dtype=param_shards_per_worker[0][name].dtype)

        for worker_idx, worker_shards in enumerate(param_shards_per_worker):
            start, end = shard_slices_per_worker[worker_idx][name]
            full_flat[start:end] = worker_shards[name]

        params[name] = full_flat.reshape(shape)

    return params

# Step 35 - zero_optimizer_step
def zero_optimizer_step(params, grads, worker_states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """Run one full ZeRO-style sharded Adam step."""
    param_shards_per_worker = []
    updated_worker_states = []

    for worker_state in worker_states:
        param_shards, updated_state = local_shard_adam_update(
            params,
            grads,
            worker_state,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )

        param_shards_per_worker.append(param_shards)
        updated_worker_states.append(updated_state)

    shapes = worker_states[0]["shapes"]
    shard_slices_per_worker = [
        worker_state["shard_slices"]
        for worker_state in worker_states
    ]

    new_params = all_gather_param_shards(
        param_shards_per_worker,
        shapes,
        shard_slices_per_worker,
    )

    return new_params, updated_worker_states

# Step 36 - compute_param_memory_bytes (not yet solved)
# TODO: implement

# Step 37 - compute_optimizer_memory_bytes (not yet solved)
# TODO: implement

# Step 38 - compute_peak_activation_memory_bytes (not yet solved)
# TODO: implement

# Step 39 - compare_memory_with_and_without_optimizations (not yet solved)
# TODO: implement

# Step 40 - full_distributed_training_loop (not yet solved)
# TODO: implement

