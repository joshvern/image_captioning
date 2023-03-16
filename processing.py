import tensorflow as tf
import re

# Text processing

strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# Image processing

def decode_and_resize(img_path, image_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def process_input(img_path, captions, image_size, vectorization):
    return decode_and_resize(img_path, image_size), vectorization(captions)

def make_dataset(images, captions, image_size, vectorization, batch_size, concurrency):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(len(images))
    dataset = dataset.map(lambda x, y: process_input(x, y, image_size, vectorization),
                          num_parallel_calls=concurrency)
    dataset = dataset.batch(batch_size).prefetch(concurrency)
    return dataset
