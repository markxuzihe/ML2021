import os
from os.path import join

# for a,b,files in os.walk('../dataset/ribfrac/train/ct'):
#     print(files)

print(os.listdir(join('../dataset/ribfrac/train', 'ct')))