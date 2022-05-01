from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from picamera.array import PiRGBArray
from picamera import PiCamera



import numpy as np
from PIL import Image
from tensorflow.python.platform import gfile
GRAPH_PB_PATH = 'output_graph.pb'

def load_graph(model_file):   #function used to load the graph
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def load_labels(label_file):  #load the labels (asphalt,dirt,grass)
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def read_tensor_from_image_file(file_name,      #take input image and transform into tensor to be fed to the model
                                input_height=224,
                                input_width=224,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  with tf.Session() as sess:
  	result = sess.run(normalized)

  return result

model_file = \
    "output_graph.pb"           
file_name = "terrain.png"
label_file = "output_labels.txt"
input_height = 224
input_width = 224
input_mean = 0
input_std = 255
input_layer = "Placeholder"
output_layer = "final_result"



###image acquisition and classification


import cv2 ##camera test capture on space

cam = PiCamera()
cam.resolution = (224,224)
rawCapture = PiRGBArray(cam, size = (224,224))

cv2.namedWindow("test")

img_counter = 0

for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True): #start capturing images
	image = frame.array
 
	# show the frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
    
	img_name = "terrain.png".format(img_counter)
	cv2.imwrite(img_name, image)
	print("{} written!".format(img_name))
	imgtest = Image.open('terrain.png').resize((224, 224))

	graph = load_graph(model_file)  #actually loading the model
	t = read_tensor_from_image_file(
	file_name,
	input_height=input_height,
	input_width=input_width,
	input_mean=input_mean,
	input_std=input_std)  

	input_name ="import/" +  input_layer
	output_name = "import/" + output_layer
	input_operation = graph.get_operation_by_name(input_name)

	output_operation = graph.get_operation_by_name(output_name) #running the classifier
	with tf.Session(graph=graph) as sess:
		sess.run(tf.global_variables_initializer())  
		results = sess.run(output_operation.outputs[0], {
    			input_operation.outputs[0]: t
	})
	results = np.squeeze(results)

	top_k = results.argsort()[-5:][::-1]
	labels = load_labels(label_file)
	for i in top_k:
		print(labels[i], results[i])

cv2.destroyAllWindows()
