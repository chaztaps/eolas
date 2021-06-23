"""MIGRATE ALL TO OBJ"""
import os,sys,shutil

target_dir = '/home/charlietapsell1989/eolas/obj'
os.makdirs(target_dir)

print('migrating...')
#gather aug folders
folders = ['/home/charlietapsell1989/eolas/augmented/deer',
           '/home/charlietapsell1989/eolas/augmented/sheep',
           '/home/charlietapsell1989/eolas/augmented/negatives']


#cycle through each aug folder and move to obj
target_dir = '/home/charlietapsell1989/eolas/obj'
for source_dir in folders:       
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
    print('done migrating',source_dir)

#cleaning .xml.txt
print('cleaning text')
for i in os.listdir(target_dir):
    if ".xml.txt" in i:
        fullpath = target_dir+i
        newpath = fullpath.replace('.xml.txt','.txt')
        os.rename(fullpath,newpath)

print('done')