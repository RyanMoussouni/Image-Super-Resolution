from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizer import Adam

class SRCNN:
    def __init__(self):
        '''
        Function defines the SRCNN model.
        
        Returns: the model itself.
        '''
        ## a sequential model
        self.model = Sequential()
        
        ## adding layers
        self.model.add(Conv2D(filters=128, kernel_size=(9,9), kernel_initializer='glorot_uniform',
                        activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                        activation='relu', padding='same', use_bias=True))
        self.model.add(Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform',
                        activation='linear', padding='valid', use_bias=True))
        
        ## define optimizer
        opt = Adam(lr=0.0003)
        
        ## compiling model
        self.model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
        pass

    def __call__(self, input):
        return self.model(input)
