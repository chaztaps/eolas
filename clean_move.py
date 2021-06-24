"""MIGRATE ALL TO OBJ"""
import shutil,os,glob,sys

target_dir = 'obj'
shutil.rmtree(target_dir, ignore_errors=True)
os.makedirs(target_dir)

print('migrating...')
#gather aug folders
folders = ['augmented/deer',
           'augmented/sheep',
           'augmented/negatives']


#cycle through each aug folder and move to obj
for source_dir in folders:       
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
    print('done migrating',source_dir)

#cleaning .xml.txt
print('cleaning text')
for path in glob.glob(target_dir+'/*.xml.txt'):
    new_path = path.replace('.xml.txt','.txt')
    os.rename(path,new_path)
print('done')
print()
