import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import config

class RibFrac_train_preprocess:
    def __init__(self, raw_dataset_path, fixed_dataset_path, args, label_path):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes = args.n_labels  # 分割类别数（只分割肝脏为2，或者分割肝脏和肿瘤为3）
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice  # 轴向外侧扩张的slice数量
        self.size = args.min_slices  # 取样的slice数量
        self.xy_down_scale = args.xy_down_scale
        self.slice_down_scale = args.slice_down_scale
        self.label_path = label_path

    def fix_data(self):
        if not os.path.exists(self.fixed_path):  # 创建保存目录
            os.makedirs(join(self.fixed_path, 'ct'))
            os.makedirs(join(self.fixed_path, 'label'))
        ct_path = self.raw_root_path
        seg_path = self.label_path
        ct_file = "RibFrac1-image.nii.gz"
        new_ct, new_seg = self.process(ct_path, seg_path, classes=self.classes)
        if new_ct != None and new_seg != None:
            sitk.WriteImage(new_ct, os.path.join(self.fixed_path, 'ct', ct_file))
            sitk.WriteImage(new_seg, os.path.join(self.fixed_path, 'label',
                                                  ct_file.replace('image', 'label')))
        # only get one example

    def process(self, ct_path, seg_path, classes=None):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        print("Ori shape:", ct_array.shape, seg_array.shape)
        if classes == 2:
            # 将标准中所有骨折处的标签融合为一个
            seg_array[seg_array > 0] = 1
        # 将CT浓度值在阈值之外的截断掉
        ct_array[ct_array > self.upper] = self.upper
        ct_array[ct_array < self.lower] = self.lower

        # # 降采样，（对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale）
        # ct_array = ndimage.zoom(ct_array,
        #                         (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
        #                         order=3)
        # seg_array = ndimage.zoom(seg_array,
        #                          (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
        #                          order=0)

        # # 找到肝脏区域开始和结束的slice，并各向外扩张
        # z = np.any(seg_array, axis=(1, 2))
        # start_slice, end_slice = np.where(z)[0][[0, -1]]
        #
        # # 两个方向上各扩张个slice
        # if start_slice - self.expand_slice < 0:
        #     start_slice = 0
        # else:
        #     start_slice -= self.expand_slice
        #
        # if end_slice + self.expand_slice >= seg_array.shape[0]:
        #     end_slice = seg_array.shape[0] - 1
        # else:
        #     end_slice += self.expand_slice
        #
        # print("Cut out range:", str(start_slice) + '--' + str(end_slice))
        # # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
        # if end_slice - start_slice + 1 < self.size:
        #     print('Too little slice，give up the sample:', ct_file)
        #     return None, None
        # # 截取保留区域
        # ct_array = ct_array[start_slice:end_slice + 1, :, :]
        # seg_array = seg_array[start_slice:end_slice + 1, :, :]
        print("Preprocessed shape:", ct_array.shape, seg_array.shape)
        # 保存为对应的格式
        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        # new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
        #                    ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        new_ct.SetSpacing((ct.GetSpacing()[0],
                           ct.GetSpacing()[1], 1))

        new_seg = sitk.GetImageFromArray(seg_array)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        # new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
        #                     ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        new_seg.SetSpacing((ct.GetSpacing()[0],
                            ct.GetSpacing()[1], 1))
        return new_ct, new_seg

if __name__ == '__main__':
    raw_dataset_path = 'E:/Data/ML_Project/dataset/ribfrac-train-images-1/Part1/RibFrac1-image.nii.gz'
    label_path = 'E:/Data/ML_Project/dataset/ribfrac-train-labels-1/Part1/RibFrac1-label.nii.gz'
    fixed_dataset_path = '../dataset/example4'

    args = config.args
    tool = RibFrac_train_preprocess(raw_dataset_path, fixed_dataset_path, args, label_path)
    tool.fix_data()  # 对原始图像进行修剪并保存