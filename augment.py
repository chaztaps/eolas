import multiprocessing as mp
mp.set_start_method('fork')
import numpy as np
import math, random, string, glob, shutil,os,sys,time
import pandas as pd
from skimage.util import random_noise
import concurrent.futures
from PIL import Image, ImageDraw,ImageOps,ImageEnhance
Image.MAX_IMAGE_PIXELS = None

#creating paths
tags = ['deer','sheep','negatives']
for t in tags:
    path = 'augmented/{}'.format(t)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

path = 'backup/'
shutil.rmtree(path, ignore_errors=True)
os.makedirs(path)

print('number of images to augment:')
INPUT = int(input())
print()

#function to retrieve coordinate data
def data(txt):
    f = open(txt, "r")
    data = f.read().split('\n')[:-1]
    return [[float(x) for x in d.split()][1:] for d in data]

#function to convert txt file to list of lists
def bbox(file):
    #open image and get size
    im = Image.open(file)
    w,h = im.size

    #extract yolo coordinate data
    coords = data((file.replace('png','xml.txt')))
    
    #extract individual coordinates for each object
    for line in coords:
        x_centre = line[0]
        y_centre = line[1]
        x_width =  line[2]
        y_width =  line[3]
    
        #convert to pixel coordinates
        dear_width = x_width*w
        dear_height = y_width*h
        X1 = (x_centre*w) - (dear_width/2)
        Y1 = (y_centre*h) - (dear_height/2)
        X2 = X1+dear_width
        Y2 = Y1+dear_height
        coords = X1,Y1,X2,Y2
    
        #draw box
        box = ImageDraw.Draw(im)
        box.rectangle((coords), outline ="white",width=1)
    return im

"""HELPER FUNCTIONS FOR THE AUGMENTATION PIPELINE"""

#Rotating a ponit (x,y) around a chosen origin
def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(-angle) * (px - ox) - math.sin(-angle) * (py - oy)
    qy = oy + math.sin(-angle) * (px - ox) + math.cos(-angle) * (py - oy)
    return qx, qy

#Converting angles to ++ quadrant for width/height adjustment
def conv(angle):
    if angle <= 90:
        return angle 
    if 90 < angle <= 180:
        return 0-angle  
    if 180 < angle <= 270:
        return angle  
    if 270 < angle <= 360:
        return 0-angle
    
#Calculating average colour of image for filling
def avg_col(im):
    R = []
    G = []
    B = []
    w,h = im.size
    for x in range(w):
        for y in range(h):
            pixels = im.getpixel((x,y))
            r,g,b = pixels[0],pixels[1],pixels[2]
            R.append(r)
            G.append(g)
            B.append(b)
    R_avg = int(sum(R)/len(R))
    G_avg = int(sum(G)/len(G))
    B_avg = int(sum(B)/len(B))
    avg_col = (R_avg,G_avg,B_avg)
    return avg_col

#angle augmentation
def angle_augment(image, CLASS, save_to):
    
    #extracting YOLO coordinates from image's associated txt file
    coords = data(image.replace('png','xml.txt'))
    
    #calculating average colour
    COL = avg_col(Image.open(image))
    
    #cycling through full 360 degrees in 5 degree increments
    for angle in range(0,360,8):
        
        #convert every YOLO line into a set of XY pixels
        new_shapes = []
        for line in coords:
            #box coordinates for non-rotated image
            im = Image.open(image)
            w,h = im.size
            x_centre = line[0]*w
            y_centre = line[1]*h
            x_width = (line[2]*w)
            y_width = (line[3]*h)
            
            #convert angles for width/height adjustment
            theta = conv(angle)
            x_width2 = (x_width*(math.cos(math.radians(theta)))) + (y_width*(math.sin(math.radians(theta))))
            y_width2 = (x_width*(math.sin(math.radians(theta)))) + (y_width*(math.cos(math.radians(theta))))
            
            #centre of non-rotated image
            ox,oy = w/2,h/2
            
            #distance of box from centre of non-rotated image
            dx,dy = ox-x_centre,oy-y_centre
           
            #rotate image and get new dimensions and centre point
            im = im.rotate(angle,expand=1,resample=3)
            w,h = im.size
            ox,oy = w/2,h/2
            
            #difference in distances between old centre and new centre
            point = ox-dx,oy-dy
            
            #perform rotation calculation on above
            R = rotate((ox,oy), point, math.radians(angle))
            a,b = R[0],R[1]
            x1,y1 = a-(x_width2/2),b-(y_width2/2)
            x2,y2 = a+(x_width2/2),b+(y_width2/2)
     
            #convert back to YOLO
            xc = abs(((x1+x2)/2)/w)
            yc = abs(((y1+y2)/2)/h)
            dx = abs(((x2-x1))/w)
            dy = abs(((y2-y1))/h)    
            new_shapes.append("{} {} {} {} {} \n".format(CLASS,xc,yc,dx,dy))
    
        #create augmented images with associated txt file
        file_name = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(20))
        im = Image.open(image).rotate(angle,expand=1,resample=3,fillcolor=COL)
        im.save('{}/{}.png'.format(save_to,file_name))
        text_file = open('{}/{}.xml.txt'.format(save_to,file_name),"w") 
        text_file.writelines(new_shapes) 
        text_file.close()   
        
        
#left-right flip augmentation
def leftright(image,CLASS,save_to):
    file_name = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(20))
    im = Image.open(image)
    im = ImageOps.mirror(im)
    txt = image.replace('.png','.xml.txt')
    lines = data(txt)
    
    new_lines = []
    for line in lines:
        x_centre = abs(1-line[0])
        y_centre = abs(line[1])
        x_width =  abs(line[2])
        y_width =  abs(line[3])
        new_line = "{} {} {} {} {} \n".format(CLASS,x_centre,y_centre,x_width,y_width)
        new_lines.append(new_line)
    text_file = open('{}/{}.xml.txt'.format(save_to,file_name),"w")
    im.save('{}/{}.png'.format(save_to,file_name))
    text_file.writelines(new_lines) 
    text_file.close()

    
#negative sample angle augmentation
def no_label_angle_aug(image, save_to):
    #calculating average colour
    COL = avg_col(Image.open(image))
    #cycling through full 360 degrees in 8 degree increments
    for angle in range(0,360,8):
        #create augmented images with associated txt file
        file_name = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(20))
        im = Image.open(image).rotate(angle,expand=1,resample=3,fillcolor=COL)
        im.save('{}/{}.png'.format(save_to,file_name))
        text_file = open('{}/{}.xml.txt'.format(save_to,file_name),"x") 
   

#negative sample flip augmentation
def no_label_leftright(image,save_to):
    file_name = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(20))
    im = Image.open(image)
    im = ImageOps.mirror(im)
    text_file = open('{}/{}.xml.txt'.format(save_to,file_name),"x")
    im.save('{}/{}.png'.format(save_to,file_name))

#fancy pca augmentation
def fancy_pca(img, alpha_std=0.1):
    orig_img = img.astype(float).copy()
    img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    alpha = np.random.normal(0, alpha_std)

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):   # RGB
        orig_img[..., idx] += add_vect[idx]

    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)
    return orig_img

#random pca/noise augmentation
def random_aug(im):
    im = Image.open(im)
    aug_type = random.choice([1,2,3])
    
    if aug_type == 1:
        con = ImageEnhance.Contrast(im)
        im = con.enhance(random.choice([0.6,0.8,1.2,1.4,1.6]))
        return im
    
    if aug_type == 2:
        return Image.fromarray(fancy_pca(np.array(im)))
    
    if aug_type == 3:
        noise_img = random_noise(np.array(im),mode='gaussian', seed=100, clip=True,var=0.005)
        noise_img = np.array(255*noise_img, dtype = 'uint8')
        return Image.fromarray(noise_img)

"""GATHERING ALL IMAGES"""
all_dirs = glob.glob('training_images/**')

train_deer = []
train_sheep = []
train_negatives = []

for d in all_dirs:
    images = glob.glob(d+'/**/*png')
    for im in images:
        if 'deer' in im:
            train_deer.append(im)
        if 'sheep' in im:
            train_sheep.append(im)
        if 'negatives' in im:
            train_negatives.append(im)
 
#displaying size of training sets
print('full training sets')           
print('deer',len(train_deer))
print('sheep',len(train_sheep))
print('negatives',len(train_negatives))
print()

#selecting some for testing
train_deer = train_deer[:INPUT]
train_sheep = train_sheep[:INPUT]
train_negatives = train_negatives[:INPUT*2]

#displaying selected training sets
print('selected training sets')           
print('deer',len(train_deer))
print('sheep',len(train_sheep))
print('negatives',len(train_negatives))
print()
#multi-core
def DEER():
    """AUGMENT DEER"""
    aug_deer = 'augmented/deer'
    #angles
    for image in train_deer:
        angle_augment(image, 0, aug_deer)
        #print('deer',len(glob.glob(aug_deer+'/*png')))
        
    #flips
    for image in glob.glob(aug_deer+'/*png'):
        leftright(image, 0, aug_deer)
        #print('deer',len(glob.glob(aug_deer+'/*png')))
 

def SHEEP():
    """AUGMENT SHEEP"""
    aug_sheep = 'augmented/sheep'
    #angles
    for image in train_sheep:
        angle_augment(image, 1, aug_sheep)
        #print('sheep',len(glob.glob(aug_sheep+'/*png')))
    
    #flips
    for image in glob.glob(aug_sheep+'/*png'):
        leftright(image, 1, aug_sheep)
        #print('sheep',len(glob.glob(aug_sheep+'/*png')))

        
def NEGS():   
    """AUGMENT NEGATIVES"""
    aug_negatives = 'augmented/negatives'
    for image in train_negatives:
        no_label_angle_aug(image, aug_negatives)
        #print('negatives',len(glob.glob(aug_negatives+'/*png')))
        
    for image in glob.glob(aug_negatives+'/*png'):
        no_label_leftright(image, aug_negatives)
        #print('negatives',len(glob.glob(aug_negatives+'/*png')))
   

print('starting augmentation...')        
start = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:
    #if __name__ == '__main__':
    f1 = executor.submit(DEER)
    f2 = executor.submit(SHEEP)
    f3 = executor.submit(NEGS)

end = time.time()

print('duration:',end-start)


"""AUGMENT RANDOM 50%"""
print('random pca/noise to half of training images...')
print()

folders = ['/augmented/deer',
           '/augmented/sheep',
           '/augmented/negatives']

for folder in folders:
    images = glob.glob(folder+'/*png')
    sample_size = int(len(images)/2)
    for image_file in random.sample(images, sample_size):
        augmented = random_aug(image_file)
        augmented.save(image_file)   

print('finished augmenting!')
