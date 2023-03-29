"""Utility functions for cell segmentation.
"""
import math
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from anndata import AnnData
from kneed import KneeLocator
from scipy import signal, sparse
from scipy.sparse import csr_matrix, issparse, spmatrix
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from skimage.segmentation import find_boundaries
from tqdm import tqdm
from typing_extensions import Literal

from .configuration import SKM
from .errors import SegmentationError
from .logging import logger_manager as lm


def circle(k: int) -> np.ndarray:
    """Draw a circle of diameter k.

    Args:
        k: Diameter

    Returns:
        8-bit unsigned integer Numpy array with 1s and 0s

    Raises:
        ValueError: if `k` is even or less than 1
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")

    r = (k - 1) // 2
    return cv2.circle(np.zeros((k, k), dtype=np.uint8), (r, r), r, 1, -1)


def knee_threshold(X: np.ndarray, n_bins: int = 256, clip: int = 5) -> float:
    """Find the knee thresholding point of an arbitrary array.

    Note:
        This function does not find the actual knee of X. It computes a
        value to be used to threshold the elements of X by finding the knee of
        the cumulative counts.

    Args:
        X: Numpy array of values
        n_bins: Number of bins to use if `X` is a float array.

    Returns:
        Knee
    """
    # Check if array only contains integers.
    _X = X.astype(int)
    if np.array_equal(X, _X):
        x = np.sort(np.unique(_X))
    else:
        x = np.linspace(X.min(), X.max(), n_bins)
    y = np.array([(X <= val).sum() for val in x]) / X.size

    x = x[clip:]
    y = y[clip:]

    kl = KneeLocator(x, y, curve="concave")
    return kl.knee


def gaussian_blur(X: np.ndarray, k: int) -> np.ndarray:
    """Gaussian blur

    This function is not designed to be called directly. Use :func:`conv2d`
    with `mode="gauss"` instead.

    Args:
        X: UMI counts per pixel.
        k: Radius of gaussian blur.

    Returns:
        Blurred array
    """
    return cv2.GaussianBlur(src=X.astype(float), ksize=(k, k), sigmaX=0, sigmaY=0)


def median_blur(X: np.ndarray, k: int) -> np.ndarray:
    """Median blur

    This function is not designed to be called directly. Use :func:`conv2d`
    with `mode="median"` instead.

    Args:
        X: UMI counts per pixel.
        k: Radius of median blur.

    Returns:
        Blurred array
    """
    return cv2.medianBlur(src=X.astype(np.uint8), ksize=k)


def conv2d(
    X: np.ndarray,
    k: int,
    mode: Literal["gauss", "median", "circle", "square"],
    bins: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convolve an array with the specified kernel size and mode.

    Args:
        X: The array to convolve.
        k: Kernel size. Must be odd.
        mode: Convolution mode. Supported modes are:
            gauss:
            circle:
            square:
        bins: Convolve per bin. Zeros are ignored.

    Returns:
        The convolved array

    Raises:
        ValueError: if `k` is even or less than 1, or if `mode` is not a
            valid mode, or if `bins` does not have the same shape as `X`
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")
    if mode not in ("median", "gauss", "circle", "square"):
        raise ValueError(f'`mode` must be one of "median", "gauss", "circle", "square"')
    if bins is not None and X.shape != bins.shape:
        raise ValueError("`bins` must have the same shape as `X`")
    if k == 1:
        return X

    def _conv(_X):
        if mode == "gauss":
            return gaussian_blur(_X, k)
        if mode == "median":
            return median_blur(_X, k)
        kernel = np.ones((k, k), dtype=np.uint8) if mode == "square" else circle(k)
        return signal.convolve2d(_X, kernel, boundary="symm", mode="same")

    if bins is not None:
        conv = np.zeros(X.shape)
        for label in np.unique(bins):
            if label > 0:
                mask = bins == label
                conv[mask] = _conv(X * mask)[mask]
        return conv
    return _conv(X)


def scale_to_01(X: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1].

    Args:
        X: Array to scale

    Returns:
        Scaled array
    """
    return (X - X.min()) / (X.max() - X.min())


def scale_to_255(X: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 255].

    Args:
        X: Array to scale

    Returns:
        Scaled array
    """
    return scale_to_01(X) * 255


def mclose_mopen(mask: np.ndarray, k: int, square: bool = False) -> np.ndarray:
    """Perform morphological close and open operations on a boolean mask.

    Args:
        X: Boolean mask
        k: Kernel size
        square: Whether or not the kernel should be square

    Returns:
        New boolean mask with morphological close and open operations performed.

    Raises:
        ValueError: if `k` is even or less than 1
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")

    kernel = np.ones((k, k), dtype=np.uint8) if square else circle(k)
    mclose = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mopen = cv2.morphologyEx(mclose, cv2.MORPH_OPEN, kernel)

    return mopen.astype(bool)


def apply_threshold(
    X: np.ndarray, k: int, threshold: Optional[Union[float, np.ndarray]] = None
) -> np.ndarray:
    """Apply a threshold value to the given array and perform morphological close
    and open operations.

    Args:
        X: The array to threshold
        k: Kernel size of the morphological close and open operations.
        threshold: Threshold to apply. By default, the knee is used.

    Returns:
        A boolean mask.
    """
    # Apply threshold and mclose,mopen
    threshold = threshold if threshold is not None else knee_threshold(X)
    print(f"threshold: {threshold}")
    mask = mclose_mopen(X >= threshold, k)
    return mask


def safe_erode(
    X: np.ndarray,
    k: int,
    square: bool = False,
    min_area: int = 1,
    n_iter: int = -1,
    float_k: Optional[int] = None,
    float_threshold: Optional[float] = None,
) -> np.ndarray:
    """Perform morphological erosion, but don't erode connected regions that
    have less than the provided area.

    Note:
        It is possible for this function to miss some small regions due to how
        erosion works. For instance, a region may have area > `min_area` which
        may be eroded in its entirety in one iteration. In this case, this
        region will not be saved.

    Args:
        X: Array to erode.
        k: Erosion kernel size
        square: Whether to use a square kernel
        min_area: Minimum area
        n_iter: Number of erosions to perform. If -1, then erosion is continued
            until every connected component is <= `min_area`.
        float_k: Morphological close and open kernel size when `X` is a
            float array.
        float_threshold: Threshold to use to determine connected components
            when `X` is a float array.

    Returns:
        Eroded array as a boolean mask

    Raises:
        ValueError: If `X` has floating point dtype but `float_threshold` is
            not provided
    """
    if X.dtype == np.dtype(bool):
        X = X.astype(np.uint8)
    is_float = np.issubdtype(X.dtype, np.floating)
    if is_float and (float_k is None or float_threshold is None):
        raise ValueError(
            "`float_k` and `float_threshold` must be provided for floating point arrays."
        )
    saved = np.zeros_like(X, dtype=bool)
    kernel = np.ones((k, k), dtype=np.uint8) if square else circle(k)

    i = 0
    with tqdm(desc="Eroding") as pbar:
        while True:
            # Find connected components and save if area <= min_area
            components = cv2.connectedComponentsWithStats(
                apply_threshold(X, float_k, float_threshold).astype(np.uint8)
                if is_float
                else X
            )

            areas = components[2][:, cv2.CC_STAT_AREA]
            for label in np.where(areas <= min_area)[0]:
                if label > 0:
                    stats = components[2][label]
                    left, top, width, height = (
                        stats[cv2.CC_STAT_LEFT],
                        stats[cv2.CC_STAT_TOP],
                        stats[cv2.CC_STAT_WIDTH],
                        stats[cv2.CC_STAT_HEIGHT],
                    )
                    saved[top : top + height, left : left + width] += (
                        components[1][top : top + height, left : left + width] == label
                    )

            X = cv2.erode(X, kernel)

            i += 1
            pbar.update(1)
            if (areas > min_area).sum() == 1 or (n_iter > 0 and n_iter == i):
                break

    mask = (X >= float_threshold) if is_float else (X > 0)
    return (mask + saved).astype(bool)


def label_overlap(X: np.ndarray, Y: np.ndarray) -> sparse.csr_matrix:
    """Compuate the overlaps between two label arrays.

    The integer labels in `X` and `Y` are used as the row and column indices
    of the resulting array.

    Note:
        The overlap array contains background overlap (index 0) as well.

    Args:
        X: First label array. Labels in this array are the rows of the resulting
            array.
        Y: Second label array. Labels in this array are the columns of the resulting
            array.

    Returns:
        A `(max(X)+1, max(Y)+1)` shape sparse array containing how many pixels for
            each label are overlapping.
    """

    def _label_overlap(X, Y):
        overlap = sparse.dok_matrix((X.max() + 1, Y.max() + 1), dtype=np.uint)
        for i in range(X.size):
            overlap[X[i], Y[i]] += 1
        return overlap

    if X.shape != Y.shape:
        raise SegmentationError(
            f"Both arrays must have the same shape, but one is {X.shape} and the other is {Y.shape}."
        )
    return _label_overlap(X.flatten(), Y.flatten()).tocsr()


def clahe(
    X: np.ndarray, clip_limit: float = 1.0, tile_grid: Tuple[int, int] = (100, 100)
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization (CLAHE).

    Args:
        X: Image to equalize
        clip_limit: Contrast clipping. Lower values retain more of homogeneous
            regions.
        tile_grid: Apply histogram equalization to tiles of this size.

    Returns:
        Equalized image
    """
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(X)


def cal_cell_area(cell_labels: np.ndarray):
    """Calculate spot numbers for each cell.

    Args:
        cell_labels: cell labels.
    Returns:
        dict
    """
    areas = {}

    t = np.bincount(cell_labels.flatten())
    for i in range(len(t)):
        if i > 0 and t[i] > 0:
            areas[i] = t[i]
    return areas


def filter_cell_labels_by_area(adata: AnnData, layer: str, area_cutoff: int = 7):
    """Filter out cells with area less than `area_cutoff`

    Args:
        adata: Input Anndata
        layer: Layer that contains UMI counts to use
        area_cutoff: cells with area less than this cutoff would be dicarded.
    """
    X = SKM.select_layer_data(adata, layer, make_dense=True)
    cells = np.unique(X)
    cells = [i for i in cells if i > 0]
    lm.main_info(f"Cell number before filtering is {len(cells)}")

    areas = cal_cell_area(X)
    filtered_cells = [k for k, v in areas.items() if v < area_cutoff]
    X = np.where(np.isin(X, filtered_cells), 0, X)
    SKM.set_layer_data(adata, layer, X)
    cells = np.unique(X)
    cells = [i for i in cells if i > 0]
    lm.main_info(f"Cell number after filtering is {len(cells)}")


def get_cell_shape(
    adata: AnnData, layer: str, thickness: int = 1, out_layer: Optional[str] = None
):
    """Set cell boundaries as 255 with thickness as `thickness`.

    Args:
        adata: Input Anndata
        layer: Layer that contains cell labels to use
        thickness: The thickness of cell boundaries
        out_layer: Layer to save results. By default, this will be `{layer}_boundary`.
    """
    labels = SKM.select_layer_data(adata, layer, make_dense=True)
    lm.main_info(f"Set cell boundaries as value of 255")

    bound = np.zeros_like(labels, dtype=np.uint8)
    for i in range(thickness):
        labels = np.where(bound == 0, labels, 0)
        bound_one = find_boundaries(labels, mode="inner").astype(np.uint8)
        bound += bound_one

    bound = bound * 255

    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.BOUNDARY_SUFFIX)
    SKM.set_layer_data(adata, out_layer, bound)


def centroids(
    bin_indices: np.ndarray, coord_min: float = 0, binsize: int = 50
) -> float:
    """Take a bin index, the mimimum coordinate and the binsize, calculate the centroid of the current bin.

    Parameters
    ----------
        bin_ind: `float`
            The bin index for the current coordinate.
        coord_min: `float`
            Minimal value for the current x or y coordinate on the entire tissue slide measured by the spatial
            transcriptomics.
        binsize: `int`
            Size of the bins to aggregate data.

    Returns
    -------
        num: `int`
            The bin index for the current coordinate.
    """
    coord_centroids = coord_min + bin_indices * binsize + binsize / 2
    return coord_centroids


def get_points_props(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate properties of labeled coordinates.

    Args:
        data: Pandas Dataframe containing `x`, `y`, `label` columns.

    Returns:
        A dataframe with properties and contours indexed by label
    """
    rows = []
    for label, _df in data.drop_duplicates(subset=["label", "x", "y"]).groupby("label"):
        points = _df[["x", "y"]].values.astype(int)
        min_offset = points.min(axis=0)
        max_offset = points.max(axis=0)
        min0, min1 = min_offset
        max0, max1 = max_offset
        hull = cv2.convexHull(points, returnPoints=True).squeeze(1)
        contour = contour_to_geo(hull)

        moments = cv2.moments(hull)
        area = moments["m00"]
        if area > 0:
            centroid0 = moments["m10"] / area
            centroid1 = moments["m01"] / area
        elif hull.shape[0] == 2:
            line = hull - min_offset
            mask = cv2.line(
                np.zeros((max_offset - min_offset + 1)[::-1], dtype=np.uint8),
                line[0],
                line[1],
                color=1,
            ).T
            area = mask.sum()
            centroid0, centroid1 = hull.mean(axis=0)
        elif hull.shape[0] == 1:
            area = 1
            centroid0, centroid1 = hull[0] + 0.5
        else:
            raise IOError(f"Convex hull contains {hull.shape[0]} points.")
        rows.append(
            [
                str(label),
                area,
                min0,
                min1,
                max0 + 1,
                max1 + 1,
                centroid0,
                centroid1,
                contour,
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "label",
            "area",
            "bbox-0",
            "bbox-1",
            "bbox-2",
            "bbox-3",
            "centroid-0",
            "centroid-1",
            "contour",
        ],
    ).set_index("label")


def get_bin_props(data: pd.DataFrame, binsize: int) -> pd.DataFrame:
    """Simulate properties of bin regions.

    Args:
        data: Pandas dataframe containing binned x, y, and cell labels.
            There should not be any duplicate cell labels.
        binsize: Bin size used

    Returns:
        A dataframe with properties and contours indexed by cell label
    """

    def create_geo(row):
        x, y = row["x"] * binsize, row["y"] * binsize
        if binsize > 1:
            geo = Polygon(
                [
                    (x, y),
                    (x + binsize, y),
                    (x + binsize, y + binsize),
                    (x, y + binsize),
                    (x, y),
                ]
            )
        else:
            geo = Point((x, y))
        geo = dumps(geo, hex=True)  # geometry object to hex
        return geo

    props = pd.DataFrame(
        {
            "label": data["label"].copy(),
            "contour": data.apply(create_geo, axis=1),
            "centroid-0": centroids(data["x"], 0, binsize),
            "centroid-1": centroids(data["y"], 0, binsize),
        }
    )
    props["area"] = binsize**2
    props["bbox-0"] = data["x"] * binsize
    props["bbox-1"] = data["y"] * binsize
    props["bbox-2"] = (data["x"] + 1) * binsize + 1
    props["bbox-3"] = (data["y"] + 1) * binsize + 1
    return props.set_index("label")


def in_concave_hull(
    p: np.ndarray, concave_hull: Union[Polygon, MultiPolygon]
) -> np.ndarray:
    """Test if points in `p` are in `concave_hull` using scipy.spatial Delaunay's find_simplex.

    Args:
        p: a `Nx2` coordinates of `N` points in `K` dimensions
        concave_hull: A polygon returned from the concave_hull function (the first value).

    Returns:

    """
    assert p.shape[1] == 2, "this function only works for two dimensional data points."

    res = [concave_hull.intersects(Point(i)) for i in p]

    return np.array(res)


def get_label_props(labels: np.ndarray) -> pd.DataFrame:
    """Measure properties of labeled cell regions.

    Args:
        labels: cell segmentation label matrix

    Returns:
        A dataframe with properties and contours indexed by label
    """

    def contour(mtx):
        """Get contours of a cell using `cv2.findContours`."""
        mtx = mtx.astype(np.uint8)
        contours = cv2.findContours(mtx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        assert len(contours) == 1
        return contours[0].squeeze(1)

    props = measure.regionprops_table(
        labels,
        properties=("label", "area", "bbox", "centroid"),
        extra_properties=[contour],
    )
    props = pd.DataFrame(props)
    props["contour"] = props.apply(
        lambda x: x["contour"] + x[["bbox-0", "bbox-1"]].to_numpy(), axis=1
    )
    props["contour"] = props["contour"].apply(contour_to_geo)
    return props.set_index(props["label"].astype(str)).drop(columns="label")


def get_coords_labels(labels: np.ndarray) -> pd.DataFrame:
    """Convert labels into sparse-format dataframe.

    Args:
        labels: cell segmentation labels matrix.

    Returns:
        A DataFrame of columns "x", "y", and "label". The coordinates are
        relative to the labels matrix.
    """
    nz = labels.nonzero()
    x, y = nz
    data = labels[nz]
    values = np.vstack((x, y, data)).T
    return pd.DataFrame(values, columns=["x", "y", "label"])


def bin_indices(coords: np.ndarray, coord_min: float, binsize: int = 50) -> int:
    """Take a DNB coordinate, the mimimum coordinate and the binsize, calculate the index of bins for the current
    coordinate.

    Parameters
    ----------
        coord: `float`
            Current x or y coordinate.
        coord_min: `float`
            Minimal value for the current x or y coordinate on the entire tissue slide measured by the spatial
            transcriptomics.
        binsize: `float`
            Size of the bins to aggregate data.

    Returns
    -------
        num: `int`
            The bin index for the current coordinate.
    """
    num = np.floor((coords - coord_min) / binsize)
    return num.astype(np.uint32)


def bin_matrix(
    X: Union[np.ndarray, spmatrix], binsize: int
) -> Union[np.ndarray, csr_matrix]:
    """Bin a matrix.

    Args:
        X: Dense or sparse matrix.
        binsize: Bin size

    Returns:
        Dense or spares matrix, depending on what the input was.
    """
    shape = (math.ceil(X.shape[0] / binsize), math.ceil(X.shape[1] / binsize))

    def _bin_sparse(X):
        nz = X.nonzero()
        x, y = nz
        data = X[nz].A.flatten()
        x_bin = bin_indices(x, 0, binsize)
        y_bin = bin_indices(y, 0, binsize)
        return csr_matrix((data, (x_bin, y_bin)), shape=shape, dtype=X.dtype)

    def _bin_dense(X):
        binned = np.zeros(shape, dtype=X.dtype)
        for x in range(X.shape[0]):
            x_bin = bin_indices(x, 0, binsize)
            for y in range(X.shape[1]):
                y_bin = bin_indices(y, 0, binsize)
                binned[x_bin, y_bin] += X[x, y]
        return binned

    if issparse(X):
        return _bin_sparse(X)
    return _bin_dense(X)
