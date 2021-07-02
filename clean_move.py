"""MIGRATE ALL TO OBJ"""
import shutil,os,glob,sys



#create obj folder within darknet/data
train_dir = 'obj'
shutil.rmtree(train_dir, ignore_errors=True)
os.makedirs(train_dir)


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



#cleaning .xml.txt
print('cleaning text')
for path in glob.glob(train_dir+'/*.xml.txt'):
    new_path = path.replace('.xml.txt','.txt')
    os.rename(path,new_path)
print('done')
print()


#moving all config shit
home = os.getcwd() 
new_home = home.replace('eolas','darknet')
shutil.move(home+'/obj.names',new_home+'/data/obj.names')
shutil.move(home+'/obj.data',new_home+'/data/obj.data')
shutil.move(home+'/CUSTOM.cfg',new_home+'/cfg/CUSTOM.cfg')
shutil.move(home+'/obj',new_home+'/data/')
shutil.move(home+'/test',new_home+'/data/')
