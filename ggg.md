Totally—let’s make a tiny **worked example** with real numbers so you can *see* how the mapping is built.

## Setup (toy numbers)

* Total anchors: `N_total = 4` → anchors indices `[0,1,2,3]`
* Visible anchors this frame (from `prefilter_voxel`): `visible_mask = [T, F, T, T]`

  * So `anchor_idx_visible = [0, 2, 3]`
  * Thus `N_vis = 3`
* Offsets per anchor: `pc.n_offsets = Koff = 2` (slots `j=0,1` per anchor)
* Per-anchor class probs (for 3 classes, Kcls=3):

  ```
  Pi_anchor (shape [4,3]) = [
    [0.7, 0.2, 0.1],  # anchor 0
    [0.1, 0.8, 0.1],  # anchor 1
    [0.3, 0.6, 0.1],  # anchor 2
    [0.4, 0.4, 0.2],  # anchor 3
  ]
  ```

## Step A — expand visible anchors to candidate gaussians

Flatten order is **anchor-major**: all offsets of anchor 0, then 2, then 3.

* `anchor_idx_visible = [0, 2, 3]`
* Repeat each visible anchor index `Koff=2` times:

  ```
  owner_idx_full = repeat_interleave → [0,0, 2,2, 3,3]    # length N_vis*Koff = 6
  ```

This says:

* candidate 0 → (anchor 0, offset 0)
* candidate 1 → (anchor 0, offset 1)
* candidate 2 → (anchor 2, offset 0)
* candidate 3 → (anchor 2, offset 1)
* candidate 4 → (anchor 3, offset 0)
* candidate 5 → (anchor 3, offset 1)

## Step B — learned gate mask over offsets

Your opacity MLP outputs `[N_vis, Koff]`. Suppose after `>0` threshold you get:

```
mask (shape [6]) = [ True, False,   True, True,   False, True ]
                    ^0     ^1        ^2    ^3      ^4     ^5
```

So active candidates are indices `{0,2,3,5}` → **4 rendered gaussians**.

## Step C — pick their owner anchors

Select from `owner_idx_full` with `mask`:

```
owner_idx = owner_idx_full[mask] = [0, 2, 2, 3]    # length = 4 == N_rendered_gaussians
```

Interpretation:

* rendered gaussian 0 came from anchor 0 (its offset 0)
* rendered gaussian 1 came from anchor 2 (offset 0)
* rendered gaussian 2 came from anchor 2 (offset 1)
* rendered gaussian 3 came from anchor 3 (offset 1)

This **ordering** matches exactly how you also filtered `xyz/color/scales/…` with the same `mask`, so all those tensors are aligned along this “rendered gaussian index”.

## Step D — map per-anchor class probs → per-rendered-gaussian probs

Index `Pi_anchor` with `owner_idx`:

```
Pi_gauss = Pi_anchor[owner_idx]
         = [
             Pi_anchor[0],  # [0.7,0.2,0.1]
             Pi_anchor[2],  # [0.3,0.6,0.1]
             Pi_anchor[2],  # [0.3,0.6,0.1]
             Pi_anchor[3],  # [0.4,0.4,0.2]
           ]
# shape [4,3] — one row per rendered gaussian, aligned with xyz/etc.
```

Now:

* `xyz.shape[0] == 4` (because you also did `xyz = xyz_all[mask]`)
* `Pi_gauss.shape[0] == 4`
* Index `g` in both refers to the **same gaussian**.

You can now do the scalar-raster trick:

* For each class `c`, set `x_g = Pi_gauss[g,c]`, render with black bg → get $\sum_g w_g(u)\, \pi_g(c)$.
* Render once with `x_g = 1` → get $\sum_g w_g(u)$.
* Normalize per-pixel to get `p_sem[1, Kcls, H, W]`.

## Debug snippet (paste near your render)

```python
# Sanity prints for one iteration
print("visible_mask.sum():", int(visible_mask.sum()))
print("anchor_idx_visible[:10]:", anchor_idx_visible[:10].tolist())
print("mask.sum() (rendered gaussians):", int(mask.view(-1).sum()))
print("xyz.shape[0]:", xyz.shape[0])

owner_idx_full = anchor_idx_visible.repeat_interleave(pc.n_offsets)
owner_idx = owner_idx_full[mask.view(-1)]
print("owner_idx.shape[0]:", owner_idx.shape[0])
print("owner_idx[:10]:", owner_idx[:10].tolist())

Pi_anchor = pc.get_sem_probs()
Pi_gauss = Pi_anchor[owner_idx]
print("Pi_gauss.shape:", tuple(Pi_gauss.shape))

assert owner_idx.shape[0] == xyz.shape[0], "Mismatch: owner_idx vs rendered gaussians"
```

If these checks pass, your mapping is correct.

---

### Key takeaways

* **`owner_idx_full`** encodes the “parent anchor” for each **candidate** (visible-anchor × offset-slot), in anchor-major order.
* **`mask`** picks which candidates became **rendered gaussians** this frame.
* **`owner_idx = owner_idx_full[mask]`** gives you the parent anchor index for **each rendered gaussian**.
* **`Pi_gauss = Pi_anchor[owner_idx]`** aligns per-gaussian semantics with your `xyz/color/...` tensors.

Once you see it with the tiny numbers above, the big tensors should feel much less mysterious.
