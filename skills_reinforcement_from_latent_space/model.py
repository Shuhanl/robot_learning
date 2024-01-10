from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

def latent_normal(inputs):
    mu, scale = inputs
    dist = tfd.Normal(loc=mu, scale=scale)
    return dist
    
def create_encoder(image_shape, proprioception_dim, layer_size=2048, sequence_length, latent_dim=256, epsilon=1e-4):

    cnn_encoder = CNN(img_height=image_shape[0], img_width=image_shape[1], img_channels=image_shape[2])
    # Image encoder
    image_input = Input(shape=(sequence_length, *image_shape))
    
    # CNN Layers
    conv = TimeDistributed(cnn_encoder)(image_input)
    
    # Proprioception feature encoder
    proprioception_input = Input(shape=(sequence_length, proprioception_dim))
    
    combined = layers.concatenate([conv, proprioception_input])

    # LSTM Layer for temporal dependencies in images
    encoded = Bidirectional(LSTM(layer_size, return_sequences=True), merge_mode='concat')(combined)
    encoded = Bidirectional(LSTM(layer_size, return_sequences=False), merge_mode='concat')(encoded)

    # Latent variable    
    mu = Dense(latent_dim, activation=None, name='mu')(encoded)
    scale = Dense(latent_dim, activation="softplus", name='sigma')(encoded + epsilon)
    mixture = tfpl.DistributionLambda(latent_normal, name='latent_variable')((mu, scale))
    
    # Define encoder model
    encoder = Model(inputs=[image_input, proprioception_input], outputs=mixture)
    return encoder
 
def create_decoder(image_shape, proprioception_dim, layer_size=2048, sequence_length, latent_dim):

    cnn_decoder = CNN(img_height=image_shape[0], img_width=image_shape[1], img_channels=image_shape[2], is_decoder=True)
    # Input #
    latent_input = Input(shape=(latent_dim,))
    # Decode to sequence
    decoded = Dense(layer_size, activation='relu')(latent_input)
    decoded = RepeatVector(sequence_length)(decoded)
    decoded = Bidirectional(LSTM(layer_size, return_sequences=True), merge_mode='concat')(decoded)
    decoded = Bidirectional(LSTM(layer_size, return_sequences=True), merge_mode='concat')(decoded)

    # Decode to images
    image_decoded = TimeDistributed(cnn_decoder)(decoded)
    
    # Decode to proprioception features
    proprioception_decoded = TimeDistributed(Dense(proprioception_dim, activation='relu'))(decoded)

    # Define decoder model
    decoder = Model(inputs=latent_input, outputs=[image_decoded, proprioception_decoded])
    return decoder


    
class spatial_softmax_cnn(tf.keras.Model):

    def __init__(self,  img_height=128, img_width = 128, img_channels=3, embedding_size=64, return_spatial_softmax = False):
        super(spatial_softmax_cnn, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.rescaling = Rescaling(1./255, input_shape=(img_height, img_width, img_channels)) # put it here for portability. Rescale pixel values from the range [0, 255] to [0, 1]
        self.conv1 = Conv2D(32, 8, strides=(4,4), padding='same', activation='relu', name='c1')
        self.conv2 = Conv2D(64, 4, strides=(2,2), padding='same', activation='relu', name='c2')
        self.conv3 = Conv2D(64, 3, strides=(1,1), padding='same', activation='relu', name='c3')
        # In between these, do a spatial softmax
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(embedding_size)
        self.return_spatial_softmax = return_spatial_softmax

         
        
    def call(self, inputs):
        x = self.rescaling(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        pre_softmax = self.conv3(x)
        
        # Assume features is of size [N, H, W, C] (batch_size, height, width, channels).
        # Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
        # jointly over the image dimensions. 
        N, H, W, C = pre_softmax.shape
        pre_softmax = tf.reshape(tf.transpose(pre_softmax, [0, 3, 1, 2]), [N * C, H * W])
        softmax = tf.nn.softmax(pre_softmax)
        # Reshape and transpose back to original format.
        softmax = tf.transpose(tf.reshape(softmax, [N, C, H, W]), [0, 2, 3, 1]) # N, H, W, C

        # Expand dims by 1
        softmax  = tf.expand_dims(softmax, -1)

        x, y = tf.range(0, W)/W, tf.range(0, H)/H # so that feature locations are on a 0-1 scale not 0-128
        X,Y = tf.meshgrid(x,y)
        # Image coords is a tensor of size [H,W,2] representing the image coordinates of each pixel
        image_coords = tf.cast(tf.stack([X,Y],-1), tf.float32)
        image_coords= tf.expand_dims(image_coords, 2)
        # multiply to get feature locations
        spatial_soft_argmax = tf.reduce_sum(softmax * image_coords, axis=[1,2])
            
        x = self.flatten(spatial_soft_argmax)
        x = self.dense1(x)
        
        return self.dense2(x), spatial_soft_argmax

 
class CNN(tf.keras.Model):
    def __init__(self, img_height=128, img_width=128, img_channels=3, embedding_size=64, is_decoder=False):
        super(ModifiedCNN, self).__init__()
        self.is_decoder = is_decoder

        # Encoder Layers
        if not is_decoder:
            self.rescaling = Rescaling(1./255, input_shape=(img_height, img_width, img_channels))
            self.conv1 = Conv2D(32, 8, strides=(4,4), padding='same', activation='relu')
            self.conv2 = Conv2D(64, 4, strides=(2,2), padding='same', activation='relu')
            self.conv3 = Conv2D(64, 3, strides=(1,1), padding='same', activation='relu')
            self.flatten = Flatten()
            self.dense1 = Dense(512, activation='relu')
            self.dense2 = Dense(embedding_size)

        # Decoder Layers
        if is_decoder:
            self.dense1_rev = Dense(512, activation='relu')
            self.dense2_rev = Dense(img_height//8 * img_width//8 * 64, activation='relu')
            self.reshape = Reshape((img_height//8, img_width//8, 64))
            self.conv3_rev = Conv2DTranspose(64, 3, strides=(1,1), padding='same', activation='relu')
            self.conv2_rev = Conv2DTranspose(64, 4, strides=(2,2), padding='same', activation='relu')
            self.conv1_rev = Conv2DTranspose(32, 8, strides=(4,4), padding='same', activation='sigmoid')
            self.rescale_back = Lambda(lambda x: x * 255)

    def call(self, inputs):
        x = self.rescaling(inputs)

        if not self.is_decoder:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dense2(x)
            return x

        if self.is_decoder:
            x = self.dense1_rev(x)
            x = self.dense2_rev(x)
            x = self.reshape(x)
            x = self.conv3_rev(x)
            x = self.conv2_rev(x)
            x = self.conv1_rev(x)
            x = self.rescale_back(x)
            return x
