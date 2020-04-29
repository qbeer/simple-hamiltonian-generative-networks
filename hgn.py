import tensorflow as tf

class HGN(tf.keras.models.Model):
    def __init__(self, phase_space_size = 128, dt = 0.1, **kwargs):
        super(HGN, self).__init__(**kwargs)
        self.phase_space_size = phase_space_size
        self.dt = dt

        self.ENCODER = self.simple_encoder()
        self.DECODER = self.simple_decoder()
        self.HNN = self.hnn()
    
    def simple_encoder(self):
        encoder = tf.keras.models.Sequential()
        encoder.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        encoder.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', strides=2))        
        
        encoder.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
        encoder.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', strides=2))

        encoder.add(tf.keras.layers.Flatten())
        encoder.add(tf.keras.layers.Dense(self.phase_space_size, activation='linear'))

        return encoder

    def simple_decoder(self):
        decoder = tf.keras.models.Sequential()
        
        decoder.add(tf.keras.layers.Dense(self.phase_space_size // 2, activation='relu'))
        decoder.add(tf.keras.layers.Reshape(target_shape=(4, 4, 4)))

        decoder.add(tf.keras.layers.UpSampling2D(interpolation='bilinear'))
        decoder.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))

        decoder.add(tf.keras.layers.UpSampling2D(interpolation='bilinear'))
        decoder.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

        decoder.add(tf.keras.layers.UpSampling2D(interpolation='bilinear'))
        decoder.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

        decoder.add(tf.keras.layers.UpSampling2D(interpolation='bilinear'))
        decoder.add(tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same'))

        return decoder

    def hnn(self):
        hnn = tf.keras.models.Sequential()

        hnn.add(tf.keras.layers.Dense(256, activation='relu'))
        hnn.add(tf.keras.layers.Dense(128, activation='relu'))
        hnn.add(tf.keras.layers.Dense(1, activation='relu'))

        return hnn

    def __call__(self, _x, training=False):
        _, height, width, channels, seq_length = _x.shape
        x = tf.reshape(_x, [1, height, width, channels * seq_length]) # concat alongside channels
        
        channels_concatenated = channels * seq_length

        encoded = self.ENCODER(x)

        q, p = tf.split(encoded, 
                        num_or_size_splits=2, axis=1)

        with tf.GradientTape() as tape:
            tape.watch(encoded)
            hamiltonian = self.HNN(encoded)
            obj = tf.reduce_mean(hamiltonian)
        
        hamiltonian_gradients = tape.gradient(obj, encoded)
        del tape

        dHdq, dHdp = tf.split(hamiltonian_gradients,
                              num_or_size_splits=2,
                              axis=1)
        
        reconstructions = [self.DECODER(q)]

        for _ in range(1, channels_concatenated // channels):
            # leap-frog update
            p_updated = p - self.dt / 2 * dHdq
            q_updated = q + self.dt * p_updated
            reconstructions.append(self.DECODER(q_updated))

        x_hat = tf.stack(reconstructions)
        x_hat = tf.transpose(x_hat, perm=[1, 2, 3, 4, 0])

        return x_hat

