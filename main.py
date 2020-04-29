from generate_data import DataGenerator
import matplotlib.pyplot as plt
import os
import random
from hgn import HGN
import tensorflow as tf
from utils import create_video
import numpy as np

datagen = DataGenerator()

trajectories = datagen.get_dataset(n_rollouts=250)

create_video(trajectories)

optimizer = tf.keras.optimizers.Adam(lr=5e-4)

hgn = HGN()
hgn.compile(optimizer=optimizer, loss='mse')

hgn.fit(trajectories, trajectories, batch_size=1, epochs=10,
        verbose=2, validation_split=0.1)

hgn.save_weights('hgn.h5')

predicted_traj = hgn(np.expand_dims(trajectories[0], axis=0))

create_video(predicted_traj)
