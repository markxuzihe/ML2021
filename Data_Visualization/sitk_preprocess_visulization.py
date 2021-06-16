import SimpleITK as sitk

# ct_path = 'E:/Data/ML_Project/dataset/ribfrac-train-images-1/Part1/RibFrac1-image.nii.gz'
#
# ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
# ct_array = sitk.GetArrayFromImage(ct)
# #print(ct_array>500)
# print(ct_array.max())
import os

dataset_path = "../dataset/example"


def load_file_name_list(file_path):
    file_name_list = []
    files = []
    for a, b, c in os.walk(file_path + '/ct'):
        files = c
    for file_name in files:
        file_name_list.append(
            [file_path + '/ct/' + file_name, file_path + '/label/' + file_name.replace('image', 'label')])
    return file_name_list

filename_list = load_file_name_list(dataset_path)

print(filename_list[0][0])
ct = sitk.ReadImage(filename_list[0][0], sitk.sitkInt16) # img

ct_array = sitk.GetArrayFromImage(ct)
print(type(ct_array))