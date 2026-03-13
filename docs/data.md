# Datasets

The `depth_estimation.data` module provides dataset classes and a unified `load_dataset()` API for loading, downloading, and iterating over standard depth estimation benchmarks.

## Quick Example

```python
from depth_estimation import load_dataset

# NYU Depth V2 — downloads ~2.8 GB on first use, then uses the local cache
ds = load_dataset("nyu_depth_v2", split="test")

sample = ds[0]
sample["pixel_values"]  # torch.Tensor (3, H, W) float32, normalised [0, 1]
sample["depth_map"]     # torch.Tensor (1, H, W) float32, metres
sample["valid_mask"]    # torch.Tensor (1, H, W) bool
```

---

## `load_dataset()` API

```python
load_dataset(
    name: str,
    split: str = "train",
    root: str = None,
    download: bool = True,
    transform = None,
    **kwargs,
) -> BaseDepthDataset
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | **required** | Dataset name. See table below. |
| `split` | `str` | `"train"` | `"train"`, `"val"`, or `"test"`. Not all datasets expose every split. |
| `root` | `str` | `None` | Local directory for the dataset. Defaults to `~/.cache/depth_estimation/datasets/<name>/`. |
| `download` | `bool` | `True` | Auto-download when files are missing (where supported). |
| `transform` | callable | `None` | Paired transform applied after tensor conversion. See [Transforms](#transforms). |
| `**kwargs` | | | Extra keyword arguments forwarded to the dataset class (e.g. `scene_type="indoors"` for DIODE). |

### Supported datasets

| `name` | Auto-download | Depth type | Scenes | Test split size |
|---|---|---|---|---|
| `nyu_depth_v2` | Yes (~2.8 GB) | Dense, metric (metres) | Indoor | 654 images |
| `diode` | Yes (~2.6 GB val / ~81 GB train) | Dense, metric (metres) | Indoor + outdoor | 771 images |
| `kitti_eigen` | No (registration required) | Sparse LiDAR, metric (metres) | Outdoor | 697 images |
| `folder` | N/A | Any | Any | N/A |

### Output schema

Every dataset returns a `dict` with three keys:

| Key | Type | Shape | Description |
|---|---|---|---|
| `pixel_values` | `torch.Tensor` float32 | `(3, H, W)` | RGB image, normalised to `[0, 1]`. |
| `depth_map` | `torch.Tensor` float32 | `(1, H, W)` | Depth in **metres** (or relative units for relative datasets). |
| `valid_mask` | `torch.Tensor` bool | `(1, H, W)` | `True` where depth is within `[min_depth, max_depth]`. |

---

## Cache location

Downloaded archives and extracted files are stored in:

```
~/.cache/depth_estimation/datasets/<name>/
```

Override with the environment variable:

```bash
export DEPTH_CACHE_DIR=/mnt/fast_ssd/datasets
```

---

## Datasets

### NYU Depth V2

Indoor RGB-D dataset captured with a Microsoft Kinect. The labeled set contains 1 449 images at 480 × 640. Depth values are in **metres**.

**Reference:** Silberman et al., "Indoor Segmentation and Support Inference from RGBD Images", ECCV 2012.

**Auto-downloads:**
- `nyu_depth_v2_labeled.mat` — all 1 449 labeled samples (~2.8 GB)
- `splits.mat` — official Eigen train/test split (795 train / 654 test)

**Requires:** `h5py` — `pip install "depth-estimation[data]"`

```python
from depth_estimation.data import NYUDepthV2Dataset

# Test set — downloads on first run
ds = NYUDepthV2Dataset(split="test")

# Specify a local root to avoid re-downloading
ds = NYUDepthV2Dataset(root="/data/nyu", split="train", download=False)

print(ds)
# NYUDepthV2Dataset(split='test', n=654, root=~/.cache/.../nyu_depth_v2)
```

```python
load_dataset("nyu_depth_v2", split="test")
```

#### Constructor arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `root` | `str \| Path` | cache dir | Directory for `.mat` files. |
| `split` | `str` | `"train"` | `"train"` (795) or `"test"` (654). |
| `transform` | callable | `None` | Paired transform. |
| `min_depth` | `float` | `1e-3` | Minimum valid depth in metres. |
| `max_depth` | `float` | `10.0` | Maximum valid depth in metres. |
| `download` | `bool` | `True` | Auto-download missing files. |

---

### DIODE

Dense indoor and outdoor depth dataset captured with a FARO Focus 3D laser scanner. Resolution is 768 × 1024. Depth values are in **metres**.

**Reference:** Vasiljevic et al., "DIODE: A Dense Indoor and Outdoor DEpth Dataset", arXiv 1908.00463, 2019.

**Auto-downloads:**
- Val set (~2.6 GB): `https://diode-dataset.s3.amazonaws.com/val.tar.gz`
- Train set (~81 GB): `https://diode-dataset.s3.amazonaws.com/train.tar.gz`

```python
from depth_estimation.data import DIODEDataset

# Val set, indoors only — downloads ~2.6 GB on first run
ds = DIODEDataset(split="val", scene_type="indoors")

# Both scene types
ds = DIODEDataset(split="val", scene_type="all")
```

```python
load_dataset("diode", split="val", scene_type="outdoors")
```

#### Constructor arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `root` | `str \| Path` | cache dir | Directory for extracted files. |
| `split` | `str` | `"val"` | `"train"`, `"val"`, or `"test"`. `"test"` loads the val set (no public GT for test). |
| `scene_type` | `str` | `"all"` | `"indoors"`, `"outdoors"`, or `"all"`. |
| `transform` | callable | `None` | Paired transform. |
| `min_depth` | `float` | `1e-3` | Minimum valid depth in metres. |
| `max_depth` | `float` | `350.0` | Maximum valid depth in metres (outdoor scenes reach ~300 m). |
| `download` | `bool` | `True` | Auto-download missing archives. |

---

### KITTI Eigen

Outdoor driving dataset with sparse LiDAR ground truth. The Eigen split is the standard benchmark for monocular depth evaluation outdoors. Depth values are in **metres**.

**Reference:** Geiger et al., "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite", CVPR 2012.

**KITTI requires registration and cannot be auto-downloaded.** Download instructions:

1. Register at <https://www.cvlibs.net/datasets/kitti/index.php>
2. Download the raw data sequences (city / residential / road):
   <https://www.cvlibs.net/datasets/kitti/raw_data.php>
3. Download the improved ground-truth depth (Garg/Eigen dense GT):
   <https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip>

The Eigen split `.txt` file lists are downloaded automatically from the [BTS repository](https://github.com/cleinc/bts).

**Expected directory layout:**

```
root/
  2011_09_26/
    2011_09_26_drive_0001_sync/
      image_02/data/
        0000000000.png
      velodyne_points/data/
        0000000000.bin
  ...
  data_depth_annotated/
    train/
      2011_09_26_drive_0001_sync/
        proj_depth/groundtruth/image_02/
          0000000005.png
    val/
      ...
  eigen_train_files_with_gt.txt
  eigen_val_files_with_gt.txt
  eigen_test_files_with_gt.txt
```

Ground-truth depth maps are 16-bit PNG files. The raw integer value divided by `256.0` gives depth in metres (KITTI convention).

```python
from depth_estimation.data import KITTIEigenDataset

ds = KITTIEigenDataset(root="/data/kitti", split="test")
```

```python
load_dataset("kitti_eigen", split="test", root="/data/kitti")
```

#### Constructor arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `root` | `str \| Path` | **required** | Path to the KITTI raw data root. |
| `split` | `str` | `"train"` | `"train"`, `"val"`, or `"test"`. |
| `filenames` | `str \| Path` | `None` | Path to a custom split `.txt` file. Auto-resolved from `root` if `None`. |
| `transform` | callable | `None` | Paired transform. |
| `min_depth` | `float` | `1e-3` | Minimum valid depth in metres. |
| `max_depth` | `float` | `80.0` | Maximum valid depth in metres. |
| `download_split` | `bool` | `True` | Auto-download the split `.txt` file. |

---

### FolderDataset

Load paired RGB + depth images from two flat directories. Files are matched by stem (filename without extension).

```python
from depth_estimation.data import FolderDataset

ds = FolderDataset(
    image_dir="data/rgb/",
    depth_dir="data/depth/",
    depth_scale=1000.0,   # millimetres → metres
)
```

```python
load_dataset(
    "folder",
    image_dir="data/rgb/",
    depth_dir="data/depth/",
    depth_scale=1000.0,
)
```

#### Supported depth formats

| Extension | How it is loaded |
|---|---|
| `.npy` | `np.load()` — values used as-is |
| `.npz` | `np.load()["depth"]` (or first key) |
| `.png` / `.tiff` | Loaded as uint16, divided by `depth_scale` |
| `.exr` | OpenCV `IMREAD_ANYDEPTH` |

#### Constructor arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `image_dir` | `str \| Path` | **required** | Directory of RGB images. |
| `depth_dir` | `str \| Path` | `None` | Directory of depth maps. If `None`, depth_map is all-zero. |
| `depth_scale` | `float` | `256.0` | Divisor for integer depth files (PNG/TIFF). `256.0` is KITTI convention; use `1000.0` for millimetres. |
| `transform` | callable | `None` | Paired transform. |
| `min_depth` | `float` | `1e-3` | Minimum valid depth. |
| `max_depth` | `float` | `1000.0` | Maximum valid depth. |

---

## Transforms

Transforms must accept and return `(pixel_values, depth_map, valid_mask)` tuples:

```python
def my_transform(pixel_values, depth_map, valid_mask):
    # pixel_values: (3, H, W) float32 tensor
    # depth_map:    (1, H, W) float32 tensor
    # valid_mask:   (1, H, W) bool tensor
    ...
    return pixel_values, depth_map, valid_mask

ds = load_dataset("nyu_depth_v2", split="train", transform=my_transform)
```

Note: Spatial transforms (flip, crop, resize) must be applied identically to all three tensors. Photometric transforms (colour jitter, brightness) apply to `pixel_values` only. See `docs/training_implementation_guide.md` for full paired transform implementations.

---

## Using with DataLoader

All dataset classes are `torch.utils.data.Dataset` subclasses and work directly with `DataLoader`.

```python
from torch.utils.data import DataLoader
from depth_estimation.data import NYUDepthV2Dataset

ds = NYUDepthV2Dataset(split="train")
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

for batch in loader:
    pixel_values = batch["pixel_values"]  # (8, 3, 480, 640)
    depth_map    = batch["depth_map"]     # (8, 1, 480, 640)
    valid_mask   = batch["valid_mask"]    # (8, 1, 480, 640)
```

> **Note:** `NYUDepthV2Dataset` opens an h5py file handle lazily per worker. Set `num_workers=0` if you encounter issues with multiprocessing and h5py on Windows.

---

## Installation

Core dataset functionality (folder, KITTI, DIODE) requires only `Pillow` and `numpy`, which are already core dependencies.

NYU Depth V2 additionally requires `h5py`:

```bash
pip install "depth-estimation[data]"
```

Or install manually:

```bash
pip install h5py tqdm
```

The `tqdm` package is optional — download progress falls back to a plain `print` when it is not installed.
