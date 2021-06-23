"""MIGRATE ALL TO OBJ"""
import os,sys,shutil

print('migrating...')
#gather aug folders
folders = ['augmented/deer',
           'augmented/sheep',
           'augmented/negatives']


#cycle through each aug folder and move to obj
target_dir = 'eolas/obj'
for source_dir in folders:       
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
    print('done migrating',source_dir)
