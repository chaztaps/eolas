import glob,os

home = os.getcwd()

#generate text file for training images
image_files = glob.glob('obj/*.png')
image_paths = [home+'/'+im for im in image_files]

with open("train.txt", "w") as outfile:
    for image in image_paths:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()

#generate text file for test images
image_files = glob.glob('test_images/*.png')
image_paths = [home+'/'+im for im in image_files]

with open("test.txt", "w") as outfile:
    for image in image_paths:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()