import tensorflow as tf
import yaml
import os
from object_detection.utils import dataset_util
import random


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

LABEL_DICT =  {
    "Green" : 1,
    "Red" : 2,
    "GreenLeft" : 3,
    "GreenRight" : 4,
    "RedLeft" : 5,
    "RedRight" : 6,
    "Yellow" : 7,
    "off" : 8,
    "RedStraight" : 9,
    "GreenStraight" : 10,
    "GreenStraightLeft" : 11,
    "GreenStraightRight" : 12,
    "RedStraightLeft" : 13,
    "RedStraightRight" : 14
    }

def create_tf_example(example):
    
    # Bosch
    #height = 720 # Image height
    #width = 1280 # Image width

    # Udacity data set
    height = 600 # Image height
    width = 800 # Image width

    filename = example['filename'] # Filename of the image. Empty if image is not from file
    filename = filename.encode()

    with tf.gfile.GFile(example['filename'], 'rb') as fid:
        encoded_image = fid.read()

    image_format = 'jpg'.encode() 

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in example['annotations']:
        #if box['occluded'] is False:
        #print("adding box")
        xmins.append(float(box['xmin'] / width))
        xmaxs.append(float((box['xmin'] + box['x_width']) / width))
        ymins.append(float(box['ymin'] / height))
        ymaxs.append(float((box['ymin']+ box['y_height']) / height))
        classes_text.append(box['class'].encode())
        classes.append(int(LABEL_DICT[box['class']]))


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    # Load annotations file
    INPUT_YAML = "object_detection/data/sim_training_data_large/sim_data_large.yaml"
    examples = yaml.load(open(INPUT_YAML, 'rb').read())

    random.shuffle(examples)
    train_examples = examples[:194]  # for training (70%)
    len_examples = len(train_examples)
    print("Loaded ", len(train_examples), "training examples")

    # Create training record
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path + ".train.record")

    for i in range(len(train_examples)):
        train_examples[i]['filename'] = os.path.abspath(os.path.join(os.path.dirname(INPUT_YAML), train_examples[i]['filename']))
    
    counter = 0
    for example in train_examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

        if counter % 10 == 0:
            print("Percent done", (counter/len_examples)*100)
        counter += 1

    writer.close()

    # Now do validation record
    val_examples = examples[194:]
    len_examples = len(val_examples)
    print("Loaded ", len(val_examples), "validation examples")

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path + ".val.record")

    for i in range(len(val_examples)):
        val_examples[i]['filename'] = os.path.abspath(os.path.join(os.path.dirname(INPUT_YAML), val_examples[i]['filename']))
    
    counter = 0
    for example in val_examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

        if counter % 10 == 0:
            print("Percent done", (counter/len_examples)*100)
        counter += 1

    writer.close()


if __name__ == '__main__':
    tf.app.run()