!pip install transformers tensorflow datasets tensorflow_addons

from datasets import load_dataset

# Load the WMT16 English-German dataset
dataset = load_dataset('wmt16', 'de-en')

# Display an example
print(dataset['train'][0])

import tensorflow as tf
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

# Preprocess the dataset for input into the model
def preprocess_data(examples):
    inputs = [f'Translate English to German: {example["en"]}' for example in examples['translation']]
    targets = [example['de'] for example in examples['translation']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length', return_tensors='tf')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length', return_tensors='tf').input_ids
    model_inputs['labels'] = labels
    decoder_inputs = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["decoder_input_ids"] = decoder_inputs["input_ids"]
    return model_inputs





# Replace the dense layers with LoRA layers
class LoRALayer(tf.keras.layers.Layer):
    def __init__(self, dense, rank=4):
        super().__init__()
        self.dense = dense
        self.rank = rank

    def build(self, input_shape):
        self.w_a = self.add_weight(shape=(input_shape[-1], self.rank),
                                   initializer='random_normal',
                                   trainable=True, name='w_a')
        self.w_b = self.add_weight(shape=(self.rank, self.dense.units),
                                   initializer='random_normal',
                                   trainable=True, name='w_b')

    def call(self, inputs):
        original_output = self.dense(inputs)
        lora_output = tf.matmul(tf.matmul(inputs, self.w_a), self.w_b)
        self.dense.trainable = False
        return original_output + lora_output




import tf_keras
import numpy as np
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense


def count_params(model):
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
    return trainable_params, non_trainable_params

# Define training configurations
ranks = [1, 4, 16]
batch_sizes = [8, 64, 128]
epochs = 2
results = {}

for rank in ranks:
    for batch_size in batch_sizes:
        print(f"Training with rank={rank}, batch_size={batch_size}")
        model = TFAutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
        model.layers[0].trainable = False
        model.layers[1].trainable = False
        model.layers[2].trainable = False
        model.layers[3] = LoRALayer(model.get_layer('lm_head'))

        # Get the number of parameters
        trainable_params, non_trainable_params = count_params(model)

        # Print the number of parameters
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {non_trainable_params}")

        # Update the batch size

        train_dataset = dataset['train'].select(range(20000)).map(preprocess_data, batched=True)
        test_dataset = dataset['test'].select(range(1000)).map(preprocess_data, batched=True)

        train_dataset =  train_dataset.to_tf_dataset(
            columns=['input_ids', 'attention_mask', 'decoder_input_ids'],
            label_cols=['labels'],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=None
        )

        test_dataset = test_dataset.to_tf_dataset(
            columns=['input_ids', 'attention_mask', 'decoder_input_ids'],
            label_cols=['labels'],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=None
        )

        # Compile the model
        model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=1e-2),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        # Train the model
        history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)
        results[(rank, batch_size)] = history.history


# Evaluate the model for each configuration
for (rank, batch_size), history in results.items():
    print(f"Results for rank={rank}, batch_size={batch_size}")
    print(history)




