"""
This code implements the rugosity measure for MNIST dataset

Rugosity  measure  quantifies  how  far  the  mappingfis  from  
          a  locally  linear  mapping  onthe data.
Higher the rugosity value, lower the smoothness of the function

For more details, Please refer the work
"Implicit Rugosity Regularization via Data Augmentation"
https://arxiv.org/pdf/1905.11639.pdf
"""
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import numpy as np
import time

inputs = keras.Input(shape=(784,), name="digits")
noise = tf.random.normal([inputs.get_shape()[1]], 0, 1, tf.float32)*0.0001
inc = tf.keras.layers.Concatenate()([inputs, inputs+noise])
x1 = layers.Dense(256, activation="relu")(inc)
x2 = layers.Dense(256, activation="relu")(x1)
x3 = layers.Dense(128, activation="relu")(x2)
outputs = layers.Dense(10, name="predictions")(x3)
model = keras.Model(inputs=inputs, outputs=outputs)

# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)



rugosity=[]
valaccList=[]
epochs = 100
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    rugosity_per_epoch=[]
   
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            tape.watch(x_batch_train)
            logits = model(x_batch_train, training=False)
            
        # Calculate A ( A[x] is a (weak) gradient of f at x)
        A = tape.gradient(logits, x_batch_train)
        # Eqn.7 and 8 from the paper ""Implicit Rugosity Regularization via Data Augmentation""
        if(x_batch_train.shape[0] == batch_size): # dont compute for last batch if size is less than defined batch size
            rg=tf.sqrt(tf.reduce_sum(tf.square(A[:32]-A[32:64])))
            rugosity_per_epoch.append(rg)

        # Calculate the gradient w.r.t loss 
        with tf.GradientTape() as tape:
            tape.watch(x_batch_train)
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
            
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                 % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * 64))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    print("Mean Rugosity over epoch: %s" % (np.array(rugosity_per_epoch).mean()))
    rugosity.append((np.array(rugosity_per_epoch).mean()))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    valaccList.append(float(val_acc))
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

