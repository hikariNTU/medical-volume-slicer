from typing import Union, Callable, List, Iterable
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


def preserve_biggest_chunks(
    vol: np.ndarray,  # Binary map for input data
    chunk_num: int = 1,  # Number of preserved chunks
    fill_hole: bool = False,  # Fill holes in output
    log: Callable = None,  # Logger function, should be callable
    spacing: List[int] = (1.0, 1.0, 1.0),  # Spacing for space calculation
) -> np.ndarray:
    # combine liver and tumor
    dilation_vol = ndimage.binary_dilation(
        vol, structure=kernel_shuttle, iterations=1  # Not sure
    ).astype(int)

    dilation_vol = measure.label(dilation_vol)
    dilation_vol *= vol  # restore?

    regions = measure.regionprops(dilation_vol)
    regions.sort(key=lambda x: x.area, reverse=True)

    if log:
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
    vol: np.ndarray = None,
    hdr=None,
    chunks: int = 2,  # num of preserved label
    merge_tumor: bool = True,
    dilation_structure=None,
):
    if isinstance(path, str):
        vol, hdr = medpy_load(path)
    assert vol and hdr, "File not load correctly? Volume or Header is NoneType"
    if merge_tumor:
        vol[vol == 2] = 1  # merge tumor
    l_vol = ndimage.binary_dilation(
        vol, structure=dilation_structure, iterations=1  # Not sure
    ).astype(int)

    l_vol = measure.label(l_vol)
    l_vol *= vol  # restore?

    regions = measure.regionprops(l_vol)
    regions.sort(key=lambda x: x.area, reverse=True)

    def test_func(x, a, b):  # linear regression
        return a + b * x

    data = []
    for i in range(2):
        if i >= len(regions):
            continue
        r = regions[i]
        chunk = {}
        chunk["id"] = f"{os.path.basename(path).split('.')[0]:05}_{i}"
        chunk["slices"] = len(r.image)
        chunk["spacing"] = hdr.spacing
        chunk["volume"] = r.area * spacing_to_cc(hdr.spacing)
        chunk["holes"] = (r.filled_area - r.area) * spacing_to_cc(hdr.spacing)
        chunk["centroid"] = f"({', '.join(f'{c:<.5}' for c in r.centroid)})"
        d_list = slice_area_change(r.image)
        params, params_covariance = (
            optimize.curve_fit(
                test_func, range(1, len(d_list) + 1), d_list, p0=[200, -3]
            )
            if len(d_list) > 2
            else ((0, 0), [(0, 0), (0, 0)])
        )
        chunk["a"] = params[0]
        chunk["b"] = params[1]
        a_err, b_err = np.sqrt(np.diag(params_covariance))
        chunk["a_err"] = a_err
        chunk["b_err"] = b_err
        chunk["pseudo_error"] = chunk["a_err"] / chunk["a"] * 100.0
        data.append(chunk)
    return data


__all__ = [slice_diff, slice_area_change, preserve_biggest_chunks, spacing_to_cc]

