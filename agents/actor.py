from keras import layers, models, optimizers, regularizers
from keras import backend as K

HIDDEN1_UNITS = 200
HIDDEN2_UNITS = 400

class Actor:
    """
    Actor (Policy) Model
    """
    def __init__(self, state_size, action_size, action_low, action_high):
        """
        Initialize parameters and build model.

        Params
        ========
            start_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """ Build an actor (policy) network that maps state -> actions. """
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        net = layers.Dense(units=HIDDEN1_UNITS, activation='relu')(states)
        net = layers.BatchNormalization()(net)

        net = layers.Dense(units=HIDDEN2_UNITS, activation='relu')(net)
        net = layers.BatchNormalization()(net)

        net = layers.Dense(units=HIDDEN1_UNITS, activation='relu')(net)

        # Add final output layer with sigmod activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name = 'raw_actions')(net)

        # add a lambda layer to return value in the action range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        # note that the loss function is defined using action value(Q value)  gradients:
        # These gradients will need to be computed using critic model, and fed in while training.
        # Hence it is specified as part of the "inputs" used in the training function
        action_gradients = layers.Input(shape=(self.action_size,))

        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        
        self.train_fn = K.function(
            inputs = [self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

