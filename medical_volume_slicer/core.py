import os
from math import isclose
from medpy import io
from medpy.core import ImageLoadingError
from skimage.transform import resize
import numpy as np
from typing import Union


class Translator:
    normal_direction = ['LR', 'AP', 'SI']
    direction_chinese = {  # convertion of direction symbol
        'S': "上頭首頂",
        'I': "下尾腳底",
        'A': "前正",
        'P': "後背反",
        'L': "左",
        'R': "右",
    }
    direction_chinese_to_symbol = {
        chinese: symbol
        for symbol, chineses in direction_chinese.items()
        for chinese in chineses
    }

    @classmethod
    def to_ch(cls, s: str) -> str:
        output = ''.join(map(lambda x: cls.direction_chinese[x][0], s))
        return output

    @classmethod
    def ch_to_dir(cls, s: str) -> str:
        output = ''.join(map(lambda x: cls.direction_chinese_to_symbol[x], s))
        return output

    @classmethod
    def to_sym(cls, direction):
        # Method 1
        # read from (0, 1, 2)
        return [
            cls.normal_direction[i] if i > 0 else cls.normal_direction[i][::-1]
            for i in direction
        ]

    @classmethod
    def from_direction_matrix(cls, matrix):
        direction = tuple(
            (i, tr[i] < 0)
            # normal_direction[i] if tr[i] > 0 else normal_direction[i][::-1]
            for tr in matrix
            for i in range(3)
            if not isclose(tr[i], 0.0)
        )
        return list(
            (cls.normal_direction[i], 'Reverse' if r else 'Normal')
            for i, r in direction
        )

    @staticmethod
    def reverse_t(transpose: tuple = (0, 1, 2)) -> tuple:
        return tuple(transpose.index(i) for i in range(len(transpose)))


class Volume:
    data: np.ndarray
    header: io.Header

    def __init__(self, datapath=None, logger=print):
        self.log = logger
        if datapath:
            self.parse_data(datapath)

        def save(path: str, force: bool = True, use_compression: bool = False):
            io.save(
                self.data,
                path,
                self.header,
                force=force,
                use_compression=use_compression,
            )

        self.save = save

    def load_data(
        self,
        arr: np.ndarray,
        direction: str = 'axial',
        spacing: tuple = (1.0, 1.0, 1.0),
    ) -> None:
        m = self.from_direction(direction=direction, reverse=True)
        arr = arr.transpose(m)
        self.data = arr
        self.header.set_voxel_spacing(spacing)

    def parse_data(self, path: str, header_only: bool = False):
        log = self.log
        if isinstance(path, str) and os.path.exists(path):
            try:
                self.data, self.header = io.load(path)
                self.datapath = path
                log(f"Data Loaded: Shape{self.data.shape}")
                log(f"path: {path}")
            except ImageLoadingError:
                log(f"Data: {path} can't be load!")
                raise ImageLoadingError
            except:
                raise
            self.direction = tuple(
                i
                # normal_direction[i] if tr[i] > 0 else normal_direction[i][::-1]
                for tr in self.header.direction
                for i in range(3)
                if tr[i] != 0.0
            )
            self.reverse = [
                tr[i] < 0
                for tr in self.header.direction
                for i in range(3)
                if tr[i] != 0.0
            ]
            self.space, self.offset, self.matrix = self.header.get_info_consistent(3)
            # 0: sagittal plane, RL
            # 1: coronal plane, AP
            # 2: axial plane, SI

    def print_direction_info(self, chinese=False):
        nd = Translator.normal_direction
        dc = Translator.direction_chinese
        if not chinese:
            print(tuple(map(lambda x: nd[x], self.direction)))
        else:
            for i, row in enumerate(map(lambda x: nd[x], self.direction)):
                if self.reverse[i]:
                    row = row[::-1]
                print(f"由{dc[row[0]][0]}至{dc[row[1]][0]}")

    def from_direction(self, direction: str = 'axial', reverse: bool = False) -> tuple:
        d = self.direction
        mapping = {
            'axial': (d[2], d[1], d[0]),
            'sagittal': (d[0], d[2], d[1]),
            'coronal': (d[1], d[2], d[0]),
        }
        try:
            m = mapping[direction]
        except KeyError:
            raise ValueError(
                f"Incorrect direction: {direction}. Expect in {{'axial', 'sagittal', 'coronal'}}"
            )
        return m if not reverse else Translator.reverse_t(m)

    def get_yield_slice(
        self,
        direction: str = 'axial',
        pad_value: Union[int, float] = -1024.0,
        generator: bool = False,
    ):
        """
        Return iterable numpy volume from this volume with different direction.

        The volume aspect ratio will be corrected by the voxel space info.
        If the shape after resizing correction is not 512, yield a slice with padding, or cropping to 512 x 512.

        Args:
            direction (str, optional): Enum in {'axial', 'sagittal', 'coronal'}. Defaults to 'axial'.
            pad_value (Union[int, float], optional): Padding value use for expanding area. Defaults to float(-1024.).

        Raises:
            ValueError: Wrapping from KeyError which indicates wrong direction mapping Enum.

        Yields:
            np.ndarray: A numpy array with shape(512, 512).
        """
        # Direction mapping.
        m = self.from_direction(direction)

        if self.data.shape[m[1]] == self.data.shape[m[2]] == 512:
            # Check if data is axial direction already
            vol = self.data.transpose(m)
        else:
            i_space = self.space[m[1]]
            j_space = self.space[m[2]]
            ratio = j_space / i_space  # Voxel Aspect ration
            scale_axis_shape = int(self.data.shape[m[1]] // ratio)
            pad_w = (
                (512 - scale_axis_shape) // 2,
                (512 - scale_axis_shape) // 2 + scale_axis_shape % 2,
            )

            vol = resize(
                self.data.transpose(m),
                (512, scale_axis_shape, 512),
                # order=1,  # Order 1 seems to break the segmentation slice.
                mode='constant',
                cval=pad_value,
                preserve_range=True,
            )

            if min(pad_w) < 0:
                vol = vol[:, -pad_w[1] : pad_w[0], :]
                # raise ValueError(f"Padding Overflow:\nPad Width: {pad_w}, from {self.datapath}, {self.header.get_info_consistent(3)}")
            else:
                vol = np.pad(vol, ((0, 0), pad_w, (0, 0)), constant_values=pad_value)
                assert vol.shape[1] == vol.shape[2] == 512, "Shape mismatch 512 x 512."

        if generator:
            def y():
                yield from vol
            return y()
        else:
            return vol


if __name__ == '__main__':
    # fmt: off
    test_mat = [
        [-0.0, 0.0, 1.0],
        [ 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ]
    # fmt: on
    print(Translator.from_direction_matrix(test_mat))
