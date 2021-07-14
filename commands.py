#uninstall cuda toolkit and all that shite
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" 
sudo apt-get --purge remove "*nvidia*"
sudo rm -rf /usr/local/cuda*
sudo apt-get update 

#reboot
sudo sudo
reboot

#get nvidia toolkit 
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run

#change the bash thing
nano ~/.bashrc
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

#reboot
sudo su 
reboot

#scp the cudnn thing across then unzip it with
tar -zxvf TAR FILE

#move all the shit to the right place
sudo mv cuda/include/* /usr/local/cuda/include/
sudo mv cuda/lib64/* /usr/local/cuda/lib64/
sudo mv cuda/NVIDIA_SLA_cuDNN_Support.txt /usr/local/cuda/

#install python shit
pip install -U scikit-image
sudo apt install libopencv-dev

#my github
git clone https://github.com/chaztaps/eolas.git

#git clone https://github.com/AlexeyAB/darknet.git
git clone https://github.com/pjreddie/darknet.git

#run augment, move+ clean
python3 augment.py
python3 clean_move.py

#generate text
python3 gen_text.py




cd darknet
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
make

wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
/home/ubuntu/darknet/data/obj/FCgfKf4WlI1labelsO6DC4w.txt

wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

./darknet detector train data/obj.data cfg/CUSTOM.cfg yolov4.conv.137 -dont_show -map