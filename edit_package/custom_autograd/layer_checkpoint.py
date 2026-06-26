"""
Layer-Granular Activation Checkpointing with Eager Deletion
===========================================================

Instead of recomputing ALL activations at once (standard checkpoint),
this recomputes one layer at a time during backward and deletes the
activation immediately after use.

Memory: O(1) intermediate activations alive at any time (instead of O(N))
Compute: O(N^2) forward passes (each layer i requires re-running layers 0..i-1)

This is the "custom autograd.Function" approach — no torch internals modified.
"""

import torch
import torch.nn as nn


class _LayerGranularFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layers_tuple, *params):
        """
        Forward pass: run all layers sequentially under no_grad,
        save only the original input x.

        Args:
            x: input tensor (may or may not require grad)
            layers_tuple: tuple of nn.Module layers (non-tensor, stored on ctx)
            *params: flattened model parameters — passed so autograd knows
                     this Function depends on differentiable tensors and
                     creates a grad_fn on the output. We do NOT use them
                     directly (they're accessed via the layers in backward).
        """
        ctx.layers = layers_tuple
        ctx.num_params = len(params)
        ctx.save_for_backward(x)

        # Stash RNG states for deterministic recomputation
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.had_cuda = torch.cuda._initialized
        if ctx.had_cuda:
            ctx.fwd_gpu_rng_state = torch.cuda.get_rng_state()

        with torch.no_grad():
            out = x
            for layer in layers_tuple:
                out = layer(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: for each layer i (in reverse order):
          1. Recompute layers 0..i-1 under no_grad to get input_i
          2. Run layer i under enable_grad to build a local autograd graph
          3. Backward through layer i → gets grad for input_i and accumulates
             param grads on layer i's parameters
          4. Delete input_i and output_i immediately
          5. Use input_i.grad as grad_output for the next (earlier) layer

        At any point, at most 1 intermediate activation is alive.
        """
        (x_saved,) = ctx.saved_tensors
        layers = ctx.layers
        n = len(layers)

        for i in range(n - 1, -1, -1):
            # ---- Step 1: recompute input to layer i ----
            # Restore RNG state so recomputation is deterministic
            rng_devices = []
            if ctx.had_cuda:
                rng_devices = list(range(torch.cuda.device_count()))

            with torch.random.fork_rng(
                devices=rng_devices,
                enabled=True,
                device_type="cuda" if ctx.had_cuda else "cpu",
            ):
                torch.set_rng_state(ctx.fwd_cpu_rng_state)
                if ctx.had_cuda:
                    torch.cuda.set_rng_state(ctx.fwd_gpu_rng_state)

                with torch.no_grad():
                    input_i = x_saved
                    for j in range(i):
                        input_i = layers[j](input_i)

            # ---- Step 2: re-run layer i WITH grad ----
            input_i = input_i.detach().requires_grad_(True)

            # Restore RNG state again and fast-forward through layers 0..i-1
            # so layer i sees the same RNG state it saw during original forward
            with torch.random.fork_rng(
                devices=rng_devices,
                enabled=True,
                device_type="cuda" if ctx.had_cuda else "cpu",
            ):
                torch.set_rng_state(ctx.fwd_cpu_rng_state)
                if ctx.had_cuda:
                    torch.cuda.set_rng_state(ctx.fwd_gpu_rng_state)

                # Fast-forward RNG through layers 0..i-1
                if i > 0:
                    with torch.no_grad():
                        _tmp = x_saved
                        for j in range(i):
                            _tmp = layers[j](_tmp)
                        del _tmp

                # Now RNG state matches what layer i saw during forward
                with torch.enable_grad():
                    output_i = layers[i](input_i)

            # ---- Step 3: backward through this single layer ----
            torch.autograd.backward(output_i, grad_output)

            # ---- Step 4: propagate gradient & cleanup ----
            grad_output = input_i.grad.detach()
            del input_i, output_i

        # grad_output is now the gradient w.r.t. the original input x
        # Return: (grad_for_x, None for layers_tuple, None for each param)
        # Param grads were already accumulated by the inner backward() calls
        return (grad_output, None) + (None,) * ctx.num_params


def layer_granular_checkpoint(layers, x):
    """
    Perform gradient checkpointing with per-layer recomputation and
    eager activation deletion.

    At any point during backward, at most ONE intermediate activation is alive.
    This is O(N^2) in compute but O(1) in activation memory.

    Args:
        layers: nn.Sequential or list/tuple of nn.Module layers
        x: input tensor

    Returns:
        Output tensor with full autograd support (param grads accumulate normally)
    """
    if isinstance(layers, nn.Sequential):
        layers = tuple(layers.children())
    elif isinstance(layers, list):
        layers = tuple(layers)

    # Collect all parameters — these are passed to apply() so that autograd
    # creates a grad_fn even when x.requires_grad is False.
    # (The actual gradient computation for params happens inside backward()
    # via the inner torch.autograd.backward() calls on per-layer sub-graphs.)
    all_params = []
    for layer in layers:
        all_params.extend(layer.parameters())

    return _LayerGranularFn.apply(x, layers, *all_params)
