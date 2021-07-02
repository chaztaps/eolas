import glob,os

home = os.getcwd()
new_home = home.replace('eolas','darknet')
os.chdir(new_home)
print(new_home)

# #generate text file for training images
image_paths = glob.glob(new_home+'/data/obj/*.png')
#image_paths = [home+'/'+im for im in image_files]

with open(new_home+"/data/train.txt", "w") as outfile:
    for image in image_paths:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()

#generate text file for test images
image_paths = glob.glob(new_home+'/data/test/*.png')
#image_paths = [home+'/'+im for im in image_files]

with open(new_home+"/data/test.txt", "w") as outfile:
    for image in image_paths:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()