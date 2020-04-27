from generate_data import DataGenerator
import matplotlib.pyplot as plt
import os
import random
from hgn import HGN
import tensorflow as tf

datagen = DataGenerator()

trajectories = datagen.get_dataset()

hgn = HGN()

optimizer = tf.keras.optimizers.Adam(lr=5e-4)

for ep in range(10):
    for traj in trajectories:
        with tf.GradientTape() as tape:
            recos = hgn(traj)
            loss = tf.reduce_mean((tf.expand_dims(traj, axis=0) - recos)**2)
        gradients = tape.gradient(loss, hgn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, hgn.trainable_variables))
        print(loss)
    hgn.save_weights('hgn.h5')


