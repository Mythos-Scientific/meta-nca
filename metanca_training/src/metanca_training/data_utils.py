import logging
import math
import os
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from jax import jit, nn, random
from PIL import Image
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

TrainInds = jax.Array
ValInds = jax.Array
Xs = jax.Array
Ys = jax.Array


def load_cifar100_arrays(
    rand_key: jax.Array,
    val_split: float = 0.2,
    nchw: bool = True,
    data_dir: str = None,
) -> tuple[Xs, Ys, TrainInds, ValInds]:
    """
    Loads CIFAR-100 into memory and returns:
      X: float32 images normalized, shape [N, C, H, W] (or [N, H, W, C] if nchw=False)
      y: one-hot labels float32 [N, 100]
      train_inds: indices for training set
      val_inds: indices for validation set
    """
    # CIFAR-100 channel means and stds (already scaled to [0,1])
    CIFAR100_MEAN = jnp.array([0.5071, 0.4867, 0.4408], dtype=jnp.float32)
    CIFAR100_STD = jnp.array([0.2675, 0.2565, 0.2761], dtype=jnp.float32)
    num_classes = 100

    # Load full train/test splits as one big batch
    train = tfds.load("cifar100", split="train", data_dir=data_dir, batch_size=-1)
    test = tfds.load("cifar100", split="test", data_dir=data_dir, batch_size=-1)

    train = tfds.as_numpy(train)
    test = tfds.as_numpy(test)

    # Concatenate train + test (if you want to shuffle/re-split)
    images = np.concatenate([train["image"], test["image"]], axis=0)  # [N, H, W, C], uint8
    labels = np.concatenate([train["label"], test["label"]], axis=0).astype(np.int32)

    # Normalize
    images = images.astype(np.float32) / 255.0
    images = (images - CIFAR100_MEAN) / CIFAR100_STD  # broadcast over channels

    if nchw:
        images = np.transpose(images, (0, 3, 1, 2))  # NCHW

    # One-hot labels
    y = np.array(jax.nn.one_hot(jnp.array(labels, dtype=jnp.int32), num_classes), dtype=np.float32)

    # Train/val split
    N = images.shape[0]
    idx = random.permutation(rand_key, jnp.arange(N))
    idx = np.array(idx)

    train_size = int((1.0 - val_split) * N)
    train_inds = idx[:train_size]
    val_inds = idx[train_size:]

    return images, y, train_inds, val_inds


def ensure_dir(directory_path: str):
    """Ensure that a directory exists; if it doesn't, create it.

    Args:
        directory_path (str): The path to the directory to check and create if necessary.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Directory created: {directory_path}")
    else:
        logger.info(f"Directory already exists: {directory_path}")


def load_image(file_path, image_size=(224, 224)):
    # Load image using PIL and resize to the desired size (e.g., 224x224 for models like ResNet)
    img = Image.open(file_path).convert("RGB")
    img = img.resize(image_size)
    img = np.array(img)  # return as np array, NHWC (H, W, C)
    return img


def normalize_images(images):
    # Normalize the data to have mean 0 and std 1 (standard normalization for most models)
    scaler = StandardScaler()
    flat_images = images.reshape(images.shape[0], -1)  # Flatten images
    normalized = scaler.fit_transform(flat_images)
    return normalized.reshape(images.shape)  # Reshape back to original image dimensions


def load_imagenet_data(
    data_dir: str, synset_mapping_file: str, image_size: Sequence[int] = (224, 224)
) -> tuple[Xs, Ys]:
    images = []
    labels = []

    max_images_per_class = 500
    # Load synset mapping
    synset_to_label = {}
    with open(synset_mapping_file, "r") as f:
        for idx, line in enumerate(f):
            synset_id, class_name = line.strip().split(" ", 1)
            synset_to_label[synset_id] = idx

    # Load train images
    total_images = 0
    for synset_id in os.listdir(os.path.join(data_dir, "train")):
        synset_path = os.path.join(data_dir, "train", synset_id)
        if synset_id not in synset_to_label:
            continue  # skip if synset not in the mapping
        for img_num, image_file in enumerate(os.listdir(synset_path)):
            if image_file.endswith(".JPEG"):
                if img_num >= max_images_per_class:
                    break
                total_images += 1
    # total_images = 600*100 # hardcoding this for now
    logger.info("Total images: " + str(total_images))
    images = np.zeros((total_images, 3, image_size[0], image_size[1]), dtype=np.uint8)
    labels = np.zeros((total_images,), dtype=int)
    logger.info("Created images array")
    i = 0
    for synset_id in os.listdir(os.path.join(data_dir, "train")):
        synset_path = os.path.join(data_dir, "train", synset_id)
        if synset_id not in synset_to_label:
            continue  # skip if synset not in the mapping

        label = synset_to_label[synset_id]
        for img_num, image_file in enumerate(os.listdir(synset_path)):
            if image_file.endswith(".JPEG"):
                if img_num >= max_images_per_class:
                    break
                img_path = os.path.join(synset_path, image_file)
                images[i, ...] = load_image(img_path, image_size)
                # images.append(img)
                # images[i, ...] = img
                labels[i, ...] = label
                i += 1
                if i % 1000 == 0:
                    logger.info("Loaded " + str(i) + " images")

    logger.info("Loaded " + str(i) + " images")
    # Convert lists to numpy arrays
    # print('Making lists into arrays')
    # images = np.array(images)
    # labels = np.array(labels)

    # Normalize the images
    # images = normalize_images(images)
    # images = images.transpose(0, 3, 1, 2)

    return images, labels


def get_imagenet_datasets(
    rand_key: jax.random.PRNGKey,
    data_dir: str,
    synset_mapping_file: str,
    image_size: Sequence[int] = (224, 224),
) -> tuple[Xs, Ys, TrainInds, ValInds]:
    # Load data
    X, y = load_imagenet_data(data_dir, synset_mapping_file, image_size)

    dataset_size = X.shape[0]
    num_classes = len(np.unique(y))  # assuming 1000 classes for ImageNet

    y = nn.one_hot(np.array(y, dtype=np.int32), num_classes)

    # Split data into training and validation sets
    shuffled_indices = random.permutation(rand_key, np.arange(dataset_size))
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_inds = shuffled_indices[:train_size]
    val_inds = shuffled_indices[train_size : train_size + val_size]

    return X, y, train_inds, val_inds


def get_linearly_separable_dataset(rand_key, dataset_size=100, train_ratio=0.8):
    # Generate random points
    X = np.random.randn(dataset_size, 2)

    # Define the decision boundary (y = x + 1)
    # Generate labels based on the decision boundary
    y = (X[:, 1] > X[:, 0] + 1).astype(int)

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse=False, categories="auto")
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Shuffle the dataset
    shuffled_indices = random.permutation(rand_key, np.arange(0, dataset_size), axis=0)

    # Split the dataset into training and validation sets
    train_size = int(train_ratio * dataset_size)
    train_inds = shuffled_indices[:train_size]
    val_inds = shuffled_indices[train_size:]

    return X, y, train_inds, val_inds


def get_iris_datasets(rand_key: jax.random.PRNGKey) -> tuple[Xs, Ys, TrainInds, ValInds]:
    iris = datasets.load_iris()
    X = iris["data"]
    dataset_size = X.shape[0]
    num_classes = 3
    y = nn.one_hot(iris["target"], num_classes)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    shuffled_indices = random.permutation(rand_key, np.arange(0, dataset_size), axis=0)
    train_inds = shuffled_indices[:train_size]
    val_inds = shuffled_indices[:val_size]
    return X, y, train_inds, val_inds


def get_mnist_datasets(
    rand_key: jax.random.PRNGKey, reshape: bool = False
) -> tuple[Xs, Ys, TrainInds, ValInds]:
    # Load MNIST data from sklearn
    mnist = datasets.fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X, y = mnist
    X = np.array(X, dtype=np.float32)  # Ensure the data type is float for scaling

    # Normalize the data to have mean 0 and std 1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # No need to reshape since we're using flat images

    dataset_size = X.shape[0]
    num_classes = 10
    if reshape:
        X = X.reshape(dataset_size, 1, 28, 28)
    y = nn.one_hot(np.array(y, dtype=np.int32), num_classes)

    # Split data into training and validation sets
    shuffled_indices = random.permutation(rand_key, np.arange(dataset_size))
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_inds = shuffled_indices[:train_size]
    val_inds = shuffled_indices[train_size : train_size + val_size]

    return X, y, train_inds, val_inds


def make_train_val_batches(rng_key: jax.random.PRNGKey, imgs_b, labs_b, mask_b, split: float = 0.8):
    n_batches = imgs_b.shape[0]
    idx = jax.random.permutation(rng_key, n_batches)
    split_pt = int(split * n_batches)
    train_idx, val_idx = idx[:split_pt], idx[split_pt:]

    train = (imgs_b[train_idx], labs_b[train_idx], mask_b[train_idx])
    val = (imgs_b[val_idx], labs_b[val_idx], mask_b[val_idx])
    return train, val


@jit
def promote_image_batch(batch_uint8):
    # NHWC uint8 -> float32 normalized to [0,1]
    if batch_uint8.dtype == jnp.uint8:
        batch = batch_uint8.astype(jnp.float32) / jnp.float32(255.0)
        return batch
    else:
        return batch_uint8


def load_imagenet_batched_sharded(
    data_dir: str,
    synset_mapping_file: str,
    rng_key,
    batch_size: int = 64,
    train_fraction: float = 0.8,
    max_per_class: int = 10000000,
    # max_per_class: int = 1,
    image_size: tuple[int, int] = (224, 224),
):
    # — 1. synset → label id ------------------------------------------------------
    syn2lbl = {}
    with open(synset_mapping_file) as f:
        for idx, line in enumerate(f):
            syn, _ = line.strip().split(" ", 1)
            syn2lbl[syn] = idx
    n_cls = len(syn2lbl)

    # — 2. first pass: count how many we’ll load ---------------------------------
    train_root = Path(data_dir) / "train"
    total = 0
    per_class_files = {}
    for spp in train_root.iterdir():
        if spp.name not in syn2lbl:
            continue
        files = sorted(p for p in spp.iterdir() if p.suffix == ".JPEG")[:max_per_class]
        per_class_files[spp.name] = files
        total += len(files)

    # batches needed
    n_train = int(math.floor(total * train_fraction))
    n_val = total - n_train
    n_train_b = math.ceil(n_train / batch_size)
    n_val_b = math.ceil(n_val / batch_size)

    # — 3. allocate final tensors -------------------------------------------------
    shp = (*image_size, 3)
    train_X = np.zeros((n_train_b, batch_size, *shp), dtype=np.uint8)
    train_y = np.zeros((n_train_b, batch_size, n_cls), dtype=np.uint8)
    train_m = np.zeros((n_train_b, batch_size, 1), dtype=bool)

    val_X = np.zeros((n_val_b, batch_size, *shp), dtype=np.uint8)
    val_y = np.zeros((n_val_b, batch_size, n_cls), dtype=np.uint8)
    val_m = np.zeros((n_val_b, batch_size, 1), dtype=bool)

    # — 4. build a deterministic random iterator over all files ------------------
    all_items = [(p, syn2lbl[p.parent.name]) for p_list in per_class_files.values() for p in p_list]
    perm = np.array(jax.random.permutation(rng_key, len(all_items)))
    train_cut = n_train

    # — 5. streaming fill (no intermediate copies) -------------------------------
    pointers = dict(train=(0, 0), val=(0, 0))  # (batch_idx, in_batch)
    buffers = dict(train=(train_X, train_y, train_m), val=(val_X, val_y, val_m))

    for k, (img_path, lbl_idx) in enumerate(np.array(all_items, dtype=object)[perm]):
        split = "train" if k < train_cut else "val"
        X, Y, M = buffers[split]
        b, j = pointers[split]

        # fill
        X[b, j] = load_image(img_path, image_size)
        Y[b, j, lbl_idx] = 1
        M[b, j, 0] = True

        # advance pointer
        j += 1
        if j == batch_size:
            b += 1
            j = 0
        pointers[split] = (b, j)

    logger.info(f"✅ loaded {total} imgs → " f"{n_train_b} train batches + {n_val_b} val batches")

    return (train_X, train_y, train_m), (val_X, val_y, val_m)


# ---------------------------------------------------------------------
# 1.  Make full‑size (N, …) → (num_batches, batch_size, …) with one pad
# ---------------------------------------------------------------------
def _batchify(arr: np.ndarray, batch_size: int):
    """Return (batched, mask) with at most one copy/pad."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    n = len(arr)
    if n == 0:
        # empty set – return 0‑length views so downstream code still works
        batched = arr.reshape(0, batch_size, *arr.shape[1:])
        mask = np.zeros((0, batch_size, 1), dtype=bool)
        return batched, mask

    pad = (-n) % batch_size  # how many to add at the end
    if pad:
        pad_width = [(0, pad)] + [(0, 0)] * (arr.ndim - 1)
        arr = np.pad(arr, pad_width, mode="constant")

    # (num_batches, batch_size, …)
    batched = arr.reshape(-1, batch_size, *arr.shape[1:])

    # build a single mask tensor; overwrite only the padded part (if any)
    mask = np.ones((batched.shape[0], batch_size, 1), dtype=bool)
    if pad:
        mask[-1, -pad:, 0] = False

    return batched, mask


# ---------------------------------------------------------------------
# 2.  Simple reshape to put the *first* axis over the local devices
# ---------------------------------------------------------------------
def _shard_for_devices(*arrays, num_devices: int):
    """Turn (N, batch, …) into (num_devices, N/num_devices, batch, …)."""

    def _reshape(a):
        if a.size == 0:
            # keep empty arrays empty
            return a.reshape(0, *a.shape[1:])
        n_batches = a.shape[0]
        if n_batches % num_devices != 0:
            raise ValueError(f"num_batches={n_batches} not divisible by num_devices={num_devices}")
        return a.reshape(num_devices, n_batches // num_devices, *a.shape[1:])

    return tuple(_reshape(a) for a in arrays)


# ---------------------------------------------------------------------
# 3.  Public helper: get ready‑to‑pmap train / val iterators
# ---------------------------------------------------------------------
def prepare_batches(X: jax.Array, y, train_idx, val_idx, batch_size="all"):
    """
    Return
        (train_X, train_y, train_mask), (val_X, val_y, val_mask)
    each shaped (n_batches, batch_size, …).
    """
    if batch_size == "all":
        bs_train = max(1, len(train_idx))
        bs_val = max(1, len(val_idx))
    else:
        bs_train = min(batch_size, max(1, len(train_idx)))
        bs_val = min(batch_size, max(1, len(val_idx)))

    # ── training set ──────────────────────────────────────────────────
    train_X_b, train_mask = _batchify(X[train_idx], bs_train)
    train_y_b, _ = _batchify(y[train_idx], bs_train)  # shape only; mask already built

    # ── validation set ───────────────────────────────────────────────
    val_X_b, val_mask = _batchify(X[val_idx], bs_val)
    val_y_b, _ = _batchify(y[val_idx], bs_val)

    return (train_X_b, train_y_b, train_mask), (val_X_b, val_y_b, val_mask)
