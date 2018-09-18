# Traffic Light detection and classifier - Gaspard Shen

Here is the simple record and more likely the step by step note of how I train the traffic light by TensorFlow object detection API and the image classifier by Keras to implement CNN.

Not very well organize now, will clean up later on.

### Training
python model_main.py --alsologtostderr --model_dir=training/ --pipeline_config_path=ssd_mobilenet_v1_coco.config --num_train_steps=10000 --num_eval_steps=1000

### Generate the inference graph
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt-51187 --output_directory traffic_light_detection_inference_py27_TF13

### Copy file to AWS
scp -r -i demo2.pem ssd_mobilenet_v1_coco.config ubuntu@ec2-54-67-22-53.us-west-1.compute.amazonaws.com:~/models/research/object_detection/

### Set up the TensorFlow object_detection
git clone https://github.com/tensorflow/models.git

### install package
apt-get install protobuf-compiler python-pil python-lxml python-tk

pip install Cython
pip install contextlib2
pip install jupyter
pip install matplotlib

### install cocoapi
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ~/models/research/

### Manual protobuf-compiler installation and usage
### From tensorflow/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip

### From tensorflow/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.

### From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

### You can test that you have correctly installed the Tensorflow Object Detection
python object_detection/builders/model_builder_test.py
