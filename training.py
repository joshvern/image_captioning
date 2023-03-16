import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from processing import decode_and_resize
from textwrap import wrap

# Learning Rate Scheduler for the optimizer
class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )

def generate_caption(caption_model, vectorization, valid_data, seq_length, image_size, samples = 10):
    valid_data = list(valid_data.items())
    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = seq_length - 1
    
    fig, axs = plt.subplots(1, samples, figsize = (40, 20))
    fig.figure(facecolor='white')
    for ax in axs.reshape(-1):
        sample = np.random.choice(range(len(valid_data)))
        sample = valid_data[sample]
        # Select a random image from the validation dataset
        sample_img = sample[0]

        # Read the image from the disk
        sample_img = decode_and_resize(sample_img, image_size)

        # Pass the image to the CNN
        img = tf.expand_dims(sample_img, 0)
        img = caption_model.cnn_model(img)

        # Pass the image features to the Transformer encoder
        encoded_img = caption_model.encoder(img, training=False)

        # Generate the caption using the Transformer decoder
        decoded_caption = "<start> "
        for i in range(max_decoded_sentence_length):
            tokenized_caption = vectorization([decoded_caption])[:, :-1]
            mask = tf.math.not_equal(tokenized_caption, 0)
            predictions = caption_model.decoder(
                tokenized_caption, encoded_img, training=False, mask=mask
            )
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = index_lookup[sampled_token_index]
            if sampled_token == " <end>":
                break
            decoded_caption += " " + sampled_token

        decoded_caption = decoded_caption.replace("<start> ", "")
        decoded_caption = decoded_caption.replace(" <end>", "").strip()
        
        real_captions = list(map(lambda x: x.replace("<start> ", "").replace(" <end>", "").strip(), sample[1]))
        real_captions[0] = '\u2022 ' + real_captions[0]
        real_captions = map(lambda x: '\n'.join(wrap(x, 40, break_long_words=False)), real_captions)
        
        decoded_caption = '\n'.join(wrap('Prediction: ' + decoded_caption, 40, break_long_words=False))
        real_captions = '\n\u2022 '.join(real_captions)
        img = sample_img.numpy().clip(0, 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(decoded_caption, fontsize = 8, fontweight="bold")
        ax.set_xlabel(real_captions, fontsize = 8)
    plt.show()
    