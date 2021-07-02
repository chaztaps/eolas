"""MIGRATE ALL TO OBJ"""
import shutil,os,glob,sys



#create obj folder within darknet/data
train_dir = 'darknet/data/obj'
shutil.rmtree(train_dir, ignore_errors=True)
os.makedirs(train_dir)



#create test folder within darknet/data
test_dir = 'darknet/data/test'
shutil.rmtree(test_dir, ignore_errors=True)
os.makedirs(test_dir)



#cycle through each aug folder and move to data/obj
print('migrating obj...')
folders = ['augmented/deer',
           'augmented/sheep',
           'augmented/negatives']

for source_dir in folders:       
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), train_dir)
    print('done migrating',source_dir)



#cycle through each aug folder and move to data/test
print('migrating test...')
file_names = os.listdir('test/')
for file_name in file_names:
    shutil.move(os.path.join(source_dir, file_name), test_dir)
print('done migrating',source_dir)



#cleaning .xml.txt
print('cleaning text')
for path in glob.glob(train_dir+'/*.xml.txt'):
    new_path = path.replace('.xml.txt','.txt')
    os.rename(path,new_path)
print('done')
print()
