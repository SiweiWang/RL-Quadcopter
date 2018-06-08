from keras import layers, models, optimizers
from keras import backend as K

HIDDEN1_UNITS = 200
HIDDEN2_UNITS = 400

class Critic:
    """Critic model"""
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ===
            state_size(int): Dimension of each state 
            action_size(int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        """ build a critic (value) network that maps (start, action) pairs -> Q-values."""

        # Define input layers
        # Note that actor model is meant to map states to actions, the critic model
        # needs to map (state,action) pairs to their Q-values
        states = layers.Input(shape = (self.state_size, ), name="states")
        actions = layers.Input(shape = (self.action_size,), name="actions")

        # State and action layers are first be processed via separate "pathways"(mini sub-network), 
        # but eventually need to be combined.
        net_states = layers.Dense(units=HIDDEN1_UNITS, activation='relu')(states)
        net_states = layers.Dense(units=HIDDEN2_UNITS, activation='relu')(net_states)

        # Add hidden layers for action pathway
        net_actions = layers.Dense(units=HIDDEN1_UNITS, activation = 'relu')(actions)
        net_actions = layers.Dense(units=HIDDEN2_UNITS, activation = 'relu')(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add final output layer to produce action value (Q value)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for traning with build-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)

        self.get_action_gradients = K.function(
            inputs = [*self.model.input, K.learning_phase()],
            outputs = action_gradients
        )