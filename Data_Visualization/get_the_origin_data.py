import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

# example_filename = 'E:/Data/ML_Project/dataset/ribfrac-train-images-1/Part1/RibFrac1-image.nii.gz'
# example_filename = 'E:/Data/ML_Project/dataset/ribfrac-train-labels-1/Part1/RibFrac1-label.nii.gz'
example_filename = '../dataset/example3/ct/RibFrac1-image.nii.gz'
img = nib.load(example_filename)
print(img)
print(img.header['db_name'])

width, height, queue = img.dataobj.shape
OrthoSlicer3D(img.dataobj).show()
exit(0)
# print(img.dataobj.shape)


# 333
num = 1
for i in range(0, queue, 17):
    # img_arr = img.dataobj[:, :, i]
    img_arr = img.dataobj[:, :, i]
    # img_arr = img_arr*100
    # print(img_arr.shape)
    # print(type(img_arr))
    plt.subplot(5, 4, num)
    plt.imshow(img_arr,cmap='gray')
    num += 1
    # print(num)
    cur_sum = img_arr.sum()
    if cur_sum != 0:
        print(cur_sum)
        print("cur_place:",i)
    if num == 21:
        break
plt.show()
