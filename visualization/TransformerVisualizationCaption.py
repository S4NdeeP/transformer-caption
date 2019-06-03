import os

import tensorflow as tf

from tensor2tensor import problems
from tensor2tensor.bin import t2t_decoder  # To register the hparams set
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.visualization import attention
from tensor2tensor.visualization import visualization
from IPython.display import Image, display

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import skimage.transform
import pickle
from PIL import Image
from subprocess import call
import matplotlib.patches as patches

import base64
import csv
import sys

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']



# Saving the objects:
def save_objects(obj):
    with open('data/objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump(obj, f)

# Getting back the objects:
def restore_objects():
    with open('data/objs.pkl') as f:  # Python 3: open(..., 'rb')
        obj = pickle.load(f)
    return obj

# PUT THE MODEL YOU WANT TO LOAD HERE!
CHECKPOINT = os.path.expanduser('data/t2t_train/image_ms_coco_tokens32k_object_position/transformer-transformer_base_single_gpu')

# HParams
problem_name = 'image_ms_coco_tokens32k'
data_dir = os.path.expanduser('data/t2t_data_caption_bottom')
model_name = "transformer"
hparams_set = "transformer_base_single_gpu"


# ## Visualization
use_bottom_up_features = True
visualizer = visualization.AttentionVisualizer(hparams_set, use_bottom_up_features, model_name, data_dir, problem_name, beam_size=4)

tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

sess = tf.train.MonitoredTrainingSession(
    checkpoint_dir=CHECKPOINT,
    save_summaries_secs=0,
)

image_file_name = 'COCO_val2014_000000076619'
image_file_parts = image_file_name.split('_')
image_id = int(image_file_parts[2])
image_path = os.path.expanduser('data/visualize/'+image_file_name+'.jpg')
bottom_up_feature_filepath = os.path.expanduser('data/tmp/t2t_datagen_caption/bottom_up_features/trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv')

if use_bottom_up_features:
    with open(bottom_up_feature_filepath, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]),
                                            dtype=np.float32).tolist()
            if image_id == item['image_id']:
                print('Found Matching object features')
                break

output_string, inp_text, out_text, att_mats = visualizer.get_vis_data_from_image(sess, image_id, image_path, item['features'], item['boxes'], use_bottom_up_features)
save_objects([output_string, inp_text, out_text, att_mats])


output_string, inp_text, out_text, att_mats = restore_objects()
print(np.shape(att_mats[0]))
print(np.shape(att_mats[1]))
print(np.shape(att_mats[2]))
print(output_string)


if use_bottom_up_features:
    img = ndimage.imread(image_path, mode='RGB')
    for l in range(4,6):
        for a in range(8):
            # Plot original image
            # img = ndimage.imread(image_path)
            plt.subplot(4, 4, 1)
            plt.imshow(img)
            plt.axis('off')

            # Plot images with attention weights
            words = out_text
            for t in range(len(words)):
                plt.subplot(4, 4, t + 2)
                words[t] = words[t].replace('_', '')
                plt.text(0, 1, '%s' % words[t], color='black', backgroundcolor='white', fontsize=8)
                plt.imshow(img)
                input_output_attn = att_mats[2]
                alp_curr = np.zeros([item['image_w'], item['image_h']])
                num_boxes = item['num_boxes']
                boxes = np.asarray(item['boxes']).reshape(-1, 4)
                for box_index in range(num_boxes):
                    for x in range(int(boxes[box_index][0]), int(boxes[box_index][2])):
                        for y in range(int(boxes[box_index][1]), int(boxes[box_index][3])):
                            alp_curr[x][y] += input_output_attn[l][0][a][t][box_index]
                alp_img = alp_curr

                #alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                plt.imshow(np.transpose(alp_img), alpha=0.5)
                plt.axis('off')
            # plt.show()
            attn_file = 'data/visualize' + str(image_file_name) \
                        + '/input_output_' + 'layer_' + str(l) + 'head' + str(a)

            directory = os.path.dirname(attn_file)
            try:
                os.stat(directory)
            except:
                os.mkdir(directory)
            plt.savefig(attn_file, format='png', dpi=1000)
    exit(0)

img = inp_text.astype(int)
for l in range(6):
    for a in range(8):
        # Plot original image
        #img = ndimage.imread(image_path)
        plt.subplot(4, 4, 1)
        plt.imshow(img)
        plt.axis('off')

        # Plot images with attention weights
        words = out_text
        for t in range(len(words)):
            plt.subplot(4, 4, t+2)
            words[t] =words[t].replace('_', '')
            plt.text(0, 1, '%s' % words[t], color='black', backgroundcolor='white', fontsize=8)
            plt.imshow(img)
            input_output_attn = att_mats[2]
            alp_curr = input_output_attn[l][0][a][t].reshape(14, 14)
            alp_img = alp_curr
            alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
            plt.imshow(alp_img, alpha=0.85)
            plt.axis('off')
        #plt.show()
        attn_file = 'data/visualize/'+ str(image_file_name) \
                    + '/input_output_' + 'layer_' + str(l) + 'head' + str(a)

        directory = os.path.dirname(attn_file)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        plt.savefig(attn_file, format='png', dpi=1000)