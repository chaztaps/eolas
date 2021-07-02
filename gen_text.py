import glob,os

home = os.getcwd()

#generate text file for training images
image_paths = glob.glob('darknet/data/obj/*.png')
#image_paths = [home+'/'+im for im in image_files]

with open("darknet/data/train.txt", "w") as outfile:
    for image in image_paths:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()

#generate text file for test images
image_paths = glob.glob('darknet/data/test/*.png')
#image_paths = [home+'/'+im for im in image_files]

with open("darknet/data/test.txt", "w") as outfile:
    for image in image_paths:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()