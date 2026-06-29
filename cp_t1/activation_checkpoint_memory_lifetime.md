# Where checkpoint deletes activation memory

This note is about the installed package in conda env `sum26`:

- `torch`: `2.12.1+cu132`
- source file: `/home/andy/anaconda3/envs/sum26/lib/python3.14/site-packages/torch/utils/checkpoint.py`
- path discussed here: non-reentrant activation checkpointing, i.e. `checkpoint(..., use_reentrant=False)`

Short answer: there is usually no `del activation_x_minus_1` in the model forward. The memory saving comes from PyTorch's saved-tensor hooks. During the original forward, checkpoint intercepts tensors that autograd would normally save and stores only a lightweight holder. During backward recomputation, it materializes the needed activation again, returns it once to autograd, then clears the handle so the recomputed activation is no longer retained by checkpoint.

## Original forward path

The public `checkpoint()` function enters the non-reentrant generator before calling your function:

```py
# checkpoint.py:513-522
gen = _checkpoint_without_reentrant_generator(...)
next(gen)                      # pre-forward logic
ret = function(*args, **kwargs) # your forward runs here
next(gen)                      # post-forward logic
```

Inside `_checkpoint_without_reentrant_generator`, PyTorch creates a `_CheckpointFrame`, saves the checkpoint inputs, then runs the original forward under `_checkpoint_hook`:

```py
# checkpoint.py:1636-1651
new_frame = _CheckpointFrame(...)
new_frame.save_inputs(*args)

with _checkpoint_hook(new_frame), forward_context:
    yield
new_frame.forward_completed = True
```

That hook is the important part. Its `pack_hook` is called whenever autograd tries to save a tensor for backward inside the checkpointed region:

```py
# checkpoint.py:1137-1147
class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame) -> None:
        def pack_hook(x):
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            ...
            return holder
```

So for your example, layer `x` can still consume activation `x-1` normally during the forward computation. But if an op inside the checkpointed region tries to save `x-1` for backward, checkpoint returns `holder` instead of saving the real tensor. After layer `x` has consumed `x-1`, if no normal Python/local/autograd reference still points to `x-1`, the activation can die by normal refcount/lifetime rules. The checkpoint code does not need an explicit `del x_minus_1`; it avoids creating the long-lived autograd saved-tensor reference in the first place.

The only tensors checkpoint intentionally keeps from the original forward are the checkpoint inputs, saved here:

```py
# checkpoint.py:826-836
def save_inputs(self, *args):
    self.saved_args = [
        _make_saved_tensor(arg, is_output=False)
        if isinstance(arg, torch.Tensor) else arg
        for arg in args
    ]

def get_inputs(self):
    return [
        arg.unpack() if isinstance(arg, SavedTensor) else arg
        for arg in self.saved_args
    ]
```

## Re-forward call path

Your `due_to_the_fact.txt` points at the recompute call. The backward-time unpack hook runs the re-forward when a saved tensor is requested:

```py
# checkpoint.py:1159-1168
if not frame.is_recomputed[gid]:
    args = frame.get_inputs()
    try:
        with _recomputation_hook(weakref.ref(frame), gid), torch.autograd.enable_grad():
            _run_fn_with_dynamo_disabled(frame.recompute_fn, *args)
    except _StopRecomputationError:
        pass
```

That calls the `recompute_fn` created in `_checkpoint_without_reentrant_generator`:

```py
# checkpoint.py:1627-1634
with (
    device_autocast_ctx,
    torch.amp.autocast("cpu", **cpu_autocast_kwargs),
    recompute_context,
    device_ctx,
    nested_fx_trace_ctx,
):
    fn(*args, **kwargs)
```

So the re-forward is just your checkpointed `fn(*args, **kwargs)` run again, with RNG/autocast/context restored.

## Where recomputed activation memory is cleared

During recomputation, `_recomputation_hook.pack_hook` catches the tensors that autograd saves in the re-forward. It stores each recomputed tensor in a weak-key dictionary:

```py
# checkpoint.py:1072-1115
class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):
    ...
    def pack_hook(x):
        x = x.detach() if x.requires_grad else x
        ...
        holder = target_frame.weak_holders[recomp_idx]()
        if holder is not None:
            holder.handles[gid] = _Handle()
            target_frame.recomputed[gid][holder.handles[gid]] = x

        if target_frame.early_stop and target_frame.recomp_counter[gid] == len(
            target_frame.weak_holders
        ):
            raise _StopRecomputationError
```

The recomputed tensor is returned to autograd from `_checkpoint_hook.unpack_hook`. Immediately after reading it from `frame.recomputed`, checkpoint clears the holder's handle:

```py
# checkpoint.py:1188-1191
_internal_assert(holder.handles[gid] in frame.recomputed[gid])
ret = frame.recomputed[gid][holder.handles[gid]]
holder.handles[gid] = None
return ret
```

This is the closest code location to "delete activation memory". The tensor is stored in `frame.recomputed[gid]`, which is a `weakref.WeakKeyDictionary`:

```py
# checkpoint.py:804-810
self.weak_holders: List[ReferenceType] = []
self.recomputed: DefaultDict[
    int, weakref.WeakKeyDictionary[_Handle, torch.Tensor]
] = defaultdict(weakref.WeakKeyDictionary)
```

The key is `holder.handles[gid]`. When `holder.handles[gid] = None` runs, checkpoint drops the holder's strong reference to that `_Handle` key. Because `frame.recomputed[gid]` is a `WeakKeyDictionary`, dropping the key removes the dictionary entry, which drops checkpoint's reference to the recomputed tensor value. After the autograd consumer is done with `ret`, no checkpoint-owned reference keeps that activation alive.

PyTorch's own comment states this lifetime rule directly:

```py
# checkpoint.py:665-682
# Rule 4. Lifetime of recomputed tensors
# Recomputed tensors are considered specific to particular invocations
# of backward and are always cleared immediately as they are unpacked
...
# Instead of packing the strong reference to the key directly, we pack
# a container object, which we manually clear as we unpack.
```

## Layer `x-1` to layer `x` mental model

For a checkpointed block like:

```txt
a0 -> layer1 -> a1 -> layer2 -> a2 -> layer3 -> a3
```

During original forward:

1. `layer2` can use `a1` to compute `a2`.
2. If autograd tries to save `a1`, `_checkpoint_hook.pack_hook` saves a `_Holder` instead of `a1`.
3. Once normal forward references to `a1` are gone, checkpoint is not keeping `a1` alive.

During backward:

1. Autograd asks to unpack the saved tensor represented by a holder.
2. `_checkpoint_hook.unpack_hook` reruns the checkpointed function.
3. `_recomputation_hook.pack_hook` captures recomputed `a1`, `a2`, etc. into `frame.recomputed`.
4. The requested tensor is returned from `frame.recomputed`.
5. `holder.handles[gid] = None` immediately clears checkpoint's weak-dict entry for that recomputed tensor.

So the logic is not "compute layer `x`, then explicitly call `del activation_x_minus_1`." It is "do not save activation `x-1` during original forward; if backward needs it later, recompute it; after unpacking the recomputed tensor once, clear the checkpoint handle so it is not retained."
