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
def mlp_forward_checkpointed(x, params):
    """Run the MLP forward pass while caching only the block input x."""
    z1 = linear_forward(x, params["W1"], params["b1"])
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, params["W2"], params["b2"])

    cache = {
        "x": x,
    }

    return z2, cache

# Step 17 - mlp_backward_checkpointed (not yet solved)
# TODO: implement

# Step 18 - estimate_checkpointing_memory_savings (not yet solved)
# TODO: implement

# Step 19 - cast_to_half_precision (not yet solved)
# TODO: implement

# Step 20 - make_master_params (not yet solved)
# TODO: implement

# Step 21 - scale_loss (not yet solved)
# TODO: implement

# Step 22 - unscale_gradients (not yet solved)
# TODO: implement

# Step 23 - has_non_finite_gradients (not yet solved)
# TODO: implement

# Step 24 - mixed_precision_step (not yet solved)
# TODO: implement

# Step 25 - shard_dataset_across_workers (not yet solved)
# TODO: implement

# Step 26 - compute_local_gradients (not yet solved)
# TODO: implement

# Step 27 - all_reduce_mean (not yet solved)
# TODO: implement

# Step 28 - ring_all_reduce_mean (not yet solved)
# TODO: implement

# Step 29 - data_parallel_train_step (not yet solved)
# TODO: implement

# Step 30 - bucket_gradients (not yet solved)
# TODO: implement

# Step 31 - init_adam_state (not yet solved)
# TODO: implement

# Step 32 - partition_optimizer_state (not yet solved)
# TODO: implement

# Step 33 - local_shard_adam_update (not yet solved)
# TODO: implement

# Step 34 - all_gather_param_shards (not yet solved)
# TODO: implement

# Step 35 - zero_optimizer_step (not yet solved)
# TODO: implement

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

