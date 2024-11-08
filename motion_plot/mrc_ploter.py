import mrcfile
import numpy as np
from scipy.ndimage import zoom
from matplotlib import pyplot as plt

def mrc_to_png_with_downsample(input_mrc, output_png):
    # 打开 MRC 文件并读取数据
    with mrcfile.open(input_mrc, permissive=True) as mrc:
        data = mrc.data

    # 取第一个切片，如果是 3D 数据
    if data.ndim == 3:
        data = data[0]

    # 判断尺寸是否大于 2000x2000，如果是则降采样 4 倍
    if data.shape[0] > 2000 and data.shape[1] > 2000:
        data = zoom(data, (0.25, 0.25), order=1)  # 使用双线性插值降采样 4 倍

    # 将数据标准化到 [0, 1] 范围
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))

    # 保存为 PNG 文件
    plt.imsave(output_png, data_normalized, cmap='gray')

# 使用示例
mrc_to_png_with_downsample('/home/peiyuan/code/pp/motion_plot/20241106_gongyicheng_0637.mrc', '/home/peiyuan/code/pp/motion_plot/20241106_gongyicheng_0637.png')
