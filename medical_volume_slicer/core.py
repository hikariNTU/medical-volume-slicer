import os
from medpy import io
from medpy.core import ImageLoadingError

def hello():
    print("hello there")

normal_direction = [
    ['L', 'R'],
    ['A', 'P'],
    ['S', 'I']
]

direction_chinese = {
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

class Volume():
    def __init__(self, datapath=None, logger=print):
        self.log = logger
        if datapath:
            self.parse_data(datapath)

    def parse_data(self, args):
        log = self.log
        if isinstance(args, str) and os.path.exists(args):
            try:
                self.data, self.header = io.load(args)
                log(f"Data Loaded: Shape{self.data.shape}")
                log(f"path: {args}")
            except ImageLoadingError:
                log(f"Data: {args} can't be load!")
                raise ImageLoadingError
            except:
                raise
            self.direction = [
                i
                # normal_direction[i] if tr[i] > 0 else normal_direction[i][::-1]
                for tr in self.header.direction
                for i in range(3)
                if tr[i] != 0.0
            ]
    
    def print_direction_info(self, chinese=False):
        if not chinese:
            print(tuple(map(lambda x: normal_direction[x], self.direction)))
        else:
            for row in map(lambda x: normal_direction[x], self.direction):
                # print(tuple(map(lambda x: direction_chinese[x][0], row)))
                print(f"由{direction_chinese[row[0]][0]}至{direction_chinese[row[1]][0]}")
            

if __name__ == '__main__':
    v = Volume(datapath=R"T:/segmentation.nii.gz")
    v.print_direction_info(chinese=True)
    print(v.direction)