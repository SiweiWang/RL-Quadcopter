from keras import layers, models, optimizers
from keras import backend as K

# Set the layer number suggested by paper:  https://pdfs.semanticscholar.org/71f2/03de1a53deae81a7707143f0ed564661e279.pdf
HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300

class Critic:
    """Critic model"""
    def __init__(self, state_size, action_size, learning_rate, decay):
        """Initialize parameters and build model.
        Params
        ===
            state_size(int): Dimension of each state 
            action_size(int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        # self.activation='tanh'
        self.activation='relu'
        self.decay = decay
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
        net_states = layers.Dense(units=HIDDEN1_UNITS, activation = self.activation)(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)

        net_states = layers.Dense(units=HIDDEN2_UNITS, activation = self.activation)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)

        net_states = layers.Dense(units=HIDDEN1_UNITS, activation = self.activation)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)
    
        # Add hidden layers for action pathway
        net_actions = layers.Dense(units=HIDDEN1_UNITS, activation = self.activation)(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.5)(net_actions)

        net_actions = layers.Dense(units=HIDDEN2_UNITS, activation = self.activation)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.5)(net_actions)

        net_actions = layers.Dense(units=HIDDEN1_UNITS, activation = self.activation)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.5)(net_actions)
    
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