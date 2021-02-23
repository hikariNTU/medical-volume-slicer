from typing import Union, Callable, List, Iterable, Any
import numpy as np
from scipy import ndimage, optimize
from skimage import measure

from medpy.io import load as medpy_load
import os


def slice_diff(vol: Union[np.ndarray, str]) -> List[int]:
    if isinstance(vol, str):
        raise NotImplementedError("Path to file is not implemented now")
    return [np.sum(np.logical_xor(vol[i], vol[i + 1])) for i in range(len(vol) - 1)]


def slice_area_change(vol: Union[np.ndarray, str]) -> List[int]:
    if isinstance(vol, str):
        raise NotImplementedError("Path to file is not implemented now")
    return [
        (np.sum(vol[i + 1, :, :]).astype(int) - np.sum(vol[i, :, :]).astype(int))
        for i in range(len(vol) - 1)
    ]


def slice_diff(vol: Union[np.ndarray, str]) -> List[int]:
    return [
        (
            np.sum(
                np.logical_and(
                    np.logical_xor(vol[i], vol[i + 1]), np.logical_not(vol[i])
                )
            ),
            np.sum(np.logical_and(np.logical_xor(vol[i], vol[i + 1]), vol[i])),
        )
        for i in range(len(vol) - 1)
    ]


__unifier = {
    'mm': 0.1,
    'cm': 1.0,
    'm': 100.0,
    'km': 100000.0,
    'in': 2.54,
    'um': 0.0001,
}


def spacing_to_cc(
    spacing: Iterable, unit: Union[str, float] = 'mm', stringify: bool = False
) -> Union[float, str]:
    """
    Get single voxel cc value from voxel spacing.

    Args:
        spacing (Iterable): The spacing provide by scaning meta header.
        unit (Union[str, float], optional): The unit used in spacing. Defaults to 'mm'.
        stringify (bool, optional): Format cc value in string. Defaults to False.

    Returns:
        Union[float, str]: A float is return, if not stringify. Else, a string indicate the cc value.
    """

    try:
        scale = (
            __unifier.get(unit.lower(), 0.1) if isinstance(unit, str) else float(unit)
        )
    except TypeError:
        print(f"[Warning] Unit should be string or a number, '{type(unit)} given'")
        scale = 0.1
    assert isinstance(scale, float)
    prod = 1
    for x in spacing:
        prod *= x * scale

    return prod if not stringify else f"{prod:.08f} cm^3"


kernel_shuttle = [
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
]


def label_regions(
    vol: np.ndarray,
    dilation_structure: np.ndarray = kernel_shuttle,
    iteration: int = 1,
) -> list:
    dilation_vol = ndimage.binary_dilation(
        vol, structure=dilation_structure, iterations=1  # Not sure
    ).astype(int)

    dilation_vol = measure.label(dilation_vol)
    dilation_vol *= vol  # restore?

    regions = measure.regionprops(dilation_vol)
    regions.sort(key=lambda x: x.area, reverse=True)
    return regions


def preserve_biggest_chunks(
    vol: np.ndarray,
    chunk_num: int = 1,
    fill_hole: bool = False,
    log: Callable = None,
    spacing: List[int] = (1.0, 1.0, 1.0),
    dilation_structure: np.ndarray = kernel_shuttle,
) -> np.ndarray:
    """
    Preserve biggest linked chunks.

    Args:
        vol (np.ndarray): Binary map for input data.
        chunk_num (int, optional): Number of preserved chunks. Defaults to 1.
        fill_hole (bool, optional): Fill holes in output. Defaults to False.
        log (Callable, optional): Logger, will be call as `log("foo")`. Defaults to None.
        spacing (List[int], optional): Spacing for volume voxel. Defaults to (1.0, 1.0, 1.0).
        dilation_structure (np.ndarray, optional): Dilation algorithms matching structure. Defaults to kernel_shuttle.

    Returns:
        np.ndarray: Merged and deleted map will be return.
    """
    # combine liver and tumor?
    regions = label_regions(vol, dilation_structure=dilation_structure, iteration=1)

    if log and Callable(log):  # Only calculate information if logger is present
        for idx, region in enumerate(regions):
            log(f"[{idx+1:=^36}]")
            log(f"{'Volume:':12} {spacing_to_cc(spacing) * region.area:<10.5f} cm3")
            log(
                f"{'Holes:':12} {spacing_to_cc(spacing) * (region.filled_area - region.area):<10.5f} cm3"
            )
            # log(f"{'Solidity:':12} {region.solidity * 100:<5.2f}%")
            log(f"{'Centroid:':12} ({', '.join(f'{c:<.5}' for c in region.centroid)})")
        log(f"{'':=^38}")

    # Preserve the nth biggest Tumor area
    preserved_vol = np.zeros_like(vol)
    for region in regions[:chunk_num]:
        preserved_vol[dilation_vol == region.label] = 1

    preserved_vol = preserved_vol * vol
    # fill the hole
    if fill_hole:
        preserved_vol = ndimage.binary_fill_holes(preserved_vol).astype(int)
    return preserved_vol


def evaluate_pred_quality(
    path: str = None,
    id_name: Any = None,
    vol: np.ndarray = None,
    hdr=None,
    chunks: int = 2,  # num of preserved label
    merge_tumor: bool = True,
    dilation_structure: np.ndarray = None,
    calc_holes: bool = False,
    note: str = None,
):
    """
    Calculate quality of a prediction label map.

    Args:
        path (str, optional): path to prediction file. If not provide, vol and hdr must be provide manually. Defaults to None.
        id_name (Any, optional): Specified the id used for chunk. If not provide, automately retrieve from path if possible.
        vol (np.ndarray, optional): volume of prediction. Defaults to None.
        hdr ([type], optional): Medpy Header, or an object which return `obj.spacing` in tuple[float, float, float]. Or spacing. Defaults to None. 
        chunks (int, optional): Max number of biggest chunk to calculate. Defaults to 2.
        dilation_structure (array_like, optional): Structure used for dilation process. Defaults to None.
        note (str, optional): Note add to each chunk. Defaults to None.

    Returns:
        List of Dict: List of Dictionary data include information chunks. Sorted by area from bigger to smaller. Chunks slices number below 2 will not be consider a valid chunk and will return a (0, 0, 999.99) linear regression result.

    Chunk Dictionary:
        id (str): File basename with labeled number of chunks, eg `pred_00018_2`
        slices (int): Slices number included in this chunk
        spacing (Tuple[float]): The space of a voxel in mm.
        volume (float): The volume in this chunk, realted to spacing.
        holes (float, optional): The holes volume inside this chunk, computation extensive.
        centroid (Tuple[float]): The centroid of chunk.
        a (float): Intercept of linear regression.
        b (float): Gradient of linear regression.
        a_err (float): Error of Intercept of linear regression.
        b_err (float): Error of Gradient of linear regression.
        pseudo_error (float): `a / a_err` or `999.99`. An simple calculation of volume error.
        note (str, optional): note provided by input argument. Might not exist.

    """
    log = print
    if isinstance(path, str):
        vol, hdr = medpy_load(path)
    assert (
        vol is not None and hdr
    ), "File not load correctly? Volume or Header is NoneType"
    if merge_tumor:
        vol[vol == 2] = 1  # merge tumor

    regions = label_regions(vol, dilation_structure=dilation_structure, iteration=1)

    def test_func(x, a, b):  # linear regression
        return a + b * x

    data = []
    for i, r in enumerate(regions[:chunks]):
        chunk = {}
        chunk["id"] = (
            f"{id_name}_{i}"
            if id_name
            else (
                f"{os.path.basename(path).split('.')[0]}_{i}" if path else f"pred_{i}"
            )
        )
        chunk["slices"] = len(r.image)
        spacing = getattr(
            hdr, 'spacing', None
        )  # ignore spacing if None of the hdr is provide
        if spacing == None:
            if (
                (isinstance(hdr, tuple) or isinstance(hdr, list))
                and len(hdr) == 3
                and all(isinstance(v, float) for v in hdr)
            ):
                spacing = hdr
            else:
                log(
                    "Cannot find spacing information from Volume Header.",
                    "Using Unit Voxel",
                )
                spacing = (1.0, 1.0, 1.0)
        chunk["spacing"] = spacing
        chunk["volume"] = r.area * spacing_to_cc(spacing)
        if calc_holes:
            chunk["holes"] = (r.filled_area - r.area) * spacing_to_cc(spacing)
        chunk["centroid"] = f"({', '.join(f'{c:<.5}' for c in r.centroid)})"
        d_list = slice_area_change(r.image)
        params, params_covariance = (  # Calculate linear regression
            optimize.curve_fit(
                test_func, range(1, len(d_list) + 1), d_list, p0=[200, -3]
            )
            if len(d_list) > 2
            else ((0, 0), [(0, 0), (0, 0)])
        )
        chunk["a"] = params[0]  # intercept
        chunk["b"] = params[1]  # gradient
        a_err, b_err = np.sqrt(np.diag(params_covariance))
        chunk["a_err"] = a_err
        chunk["b_err"] = b_err
        chunk["pseudo_error"] = (
            (chunk["a_err"] / chunk["a"] * 100.0) if chunk['a'] > 1.0 else 999.99
        )  # Protect for zero-like division error
        if note:
            chunk["note"] = note
        data.append(chunk)
    return data


__all__ = [
    slice_diff,
    slice_area_change,
    preserve_biggest_chunks,
    spacing_to_cc,
    evaluate_pred_quality,
]

