import os, sys
path = "C:/Users/Vincent van Vliet/Desktop/TU/Y2/Q2/IP/Docker_ImageProcessing-updated/Docker_ImageProcessing/Labeling/labels"
dirs = os.listdir(path)
i=0
sorted = dirs.sort()
print(sorted)
for item in dirs:
   print(item)