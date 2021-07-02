#install python shit
pip install -U scikit-image
sudo apt install libopencv-dev

#my github
git clone https://github.com/chaztaps/eolas.git


#run augment, move+ clean
python3 augment.py
python3 clean_move.py


#weird link
https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-linux-x64-v7.6.5.32.tgz?DD9iWnrxBaNRNcJ8MT0nUWSHnvkajaGynzrfJSMf2eonYu4Z-cHaVXcNfAwZN2J865XWqn15RxCxCpwIIklKbeHmdw2fDbBy1kSpPffU6zHQlwM80vAgHGvM3CXdKChkOeIgVudJTUK3IQyBHlLVsLkUySfMjrnKgglavTqsdL6StI3ltp9ZO5IGM8qqw_1FAVJiXfQ0C-ktU16O0KeUvJCtjsPibSgY8A

#generate text
python3 gen_text.py

git clone https://github.com/AlexeyAB/darknet



cd darknet
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
make

wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

cp -r /home/ubuntu/eolas/obj /home/ubuntu/eolas/darknet/data
cp -r //home/ubuntu/eolas/test_images /home/ubuntu/eolas/darknet/data
cp /home/ubuntu/eolas/CUSTOM.cfg /home/ubuntu/eolas/darknet/cfg
cp /home/ubuntu/eolas/obj.names /home/ubuntu/eolas/darknet/data
cp /home/ubuntu/eolas/obj.data /home/ubuntu/eolas/darknet/data
cp /home/ubuntu/eolas/train.txt /home/ubuntu/eolas/darknet/data
cp /home/ubuntu/eolas/test.txt /home/ubuntu/eolas/darknet/data

wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

./darknet detector train data/obj.data cfg/CUSTOM.cfg yolov4.conv.137 -dont_show -map