import tensorflow as tf
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



class LSTM_Network:
    '''
    A neural network class that contains the network parameters.

    Attributes:
    -----------
    batch_size : int
        The number of samples in each mini-batch during training.
    hidden_layer : int
        The number of neurons in the hidden layer.
    clip_margin : int
        The threshold to clip gradients to prevent exploding gradients.
    learning_rate : float
        The step size used to update the weights during backpropagation.
    epochs : int
        The number of times to iterate over the entire training set.
    '''
    
    def __init__(self, batch_size: int, window_size: int, hidden_layer: int, learning_rate: float, epochs: int, clip_margin: int = 4):
        self.batch_size = batch_size
        self.window_size = window_size
        self.hidden_layer = hidden_layer
        self.clip_margin = clip_margin
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.window_size, 1])
        self.targets = tf.placeholder(tf.float32, [self.batch_size, 1])

        self.initiate_weights_n_biases()


    def train(self, X_train, y_train, verbose=False):
        self.outputs = []

        for i in range(self.batch_size):
            batch_state = np.zeros([1, self.hidden_layer], dtype=np.float32) 
            batch_output = np.zeros([1, self.hidden_layer], dtype=np.float32)
            
            for j in range(self.window_size):
                batch_state, batch_output = self.LSTM_cell(tf.reshape(self.inputs[i][j], (-1, 1)), batch_state, batch_output)
                
            self.outputs.append(tf.matmul(batch_output, self.weights_output) + self.bias_output_layer)

        loss = self.calculate_loss(self.outputs, self.targets)

        #we define optimizer with gradient clipping
        gradients = tf.gradients(loss, tf.trainable_variables())
        clipped, _ = tf.clip_by_global_norm(gradients, self.clip_margin)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        for i in range(self.epochs):
            self.traind_scores = []
            j = 0
            epoch_loss = []

            while (j + self.batch_size) <= len(X_train):
                X_batch = X_train[j : j + self.batch_size]
                y_batch = y_train[j : j + self.batch_size]
                
                o, c, _ = self.session.run([self.outputs, loss, trained_optimizer], feed_dict={self.inputs:X_batch, self.targets:y_batch})
                
                epoch_loss.append(c)
                self.traind_scores.append(o)
                j += self.batch_size

            if verbose:
                if i % (self.epochs // 4) == 0:
                    print('Epoch {}/{}'.format(i, self.epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))

        return np.mean(epoch_loss)


    def test(self, X_test):
        tests = []
        i = 0

        while i + self.batch_size <= len(X_test): 
            o = self.session.run([self.outputs], feed_dict={self.inputs:X_test[i : i + self.batch_size]})
            i += self.batch_size
            tests.append(o)
        
        tests_new = []
        for i in range(len(tests)):
            for j in range(len(tests[i][0])):
                tests_new.append(tests[i][0][j])
        
        test_results = []
        for i in range(1264):
            if i >= 1019:
                test_results.append(tests_new[i-1019])
            else:
                test_results.append(None)
        
        return test_results
    

    def calculate_loss(self, outputs, targets):
        losses = []

        for i in range(len(outputs)):
            losses.append(tf.losses.mean_squared_error(tf.reshape(targets[i], (-1, 1)), outputs[i]))
            
        loss = tf.reduce_mean(losses)

        return loss
        
        
    def LSTM_cell(self, input, output, state):
        
        input_gate = tf.sigmoid(tf.matmul(input, self.weights_input_gate) + tf.matmul(output, self.weights_input_hidden) + self.bias_input)
        
        forget_gate = tf.sigmoid(tf.matmul(input, self.weights_forget_gate) + tf.matmul(output, self.weights_forget_hidden) + self.bias_forget)
        
        output_gate = tf.sigmoid(tf.matmul(input, self.weights_output_gate) + tf.matmul(output, self.weights_output_hidden) + self.bias_output)
        
        memory_cell = tf.tanh(tf.matmul(input, self.weights_memory_cell) + tf.matmul(output, self.weights_memory_cell_hidden) + self.bias_memory_cell)
        
        state = state * forget_gate + input_gate * memory_cell
        
        output = output_gate * tf.tanh(state)
        
        return state, output
        

    def initiate_weights_n_biases(self):
        # weights for the input gate
        self.weights_input_gate = tf.Variable(tf.truncated_normal([1, self.hidden_layer], stddev=0.05))
        self.weights_input_hidden = tf.Variable(tf.truncated_normal([self.hidden_layer, self.hidden_layer], stddev=0.05))
        self.bias_input = tf.Variable(tf.zeros([self.hidden_layer]))

        # weights for the forgot gate
        self.weights_forget_gate = tf.Variable(tf.truncated_normal([1, self.hidden_layer], stddev=0.05))
        self.weights_forget_hidden = tf.Variable(tf.truncated_normal([self.hidden_layer, self.hidden_layer], stddev=0.05))
        self.bias_forget = tf.Variable(tf.zeros([self.hidden_layer]))

        # weights for the output gate
        self.weights_output_gate = tf.Variable(tf.truncated_normal([1, self.hidden_layer], stddev=0.05))
        self.weights_output_hidden = tf.Variable(tf.truncated_normal([self.hidden_layer, self.hidden_layer], stddev=0.05))
        self.bias_output = tf.Variable(tf.zeros([self.hidden_layer]))

        # weights for the memory cell
        self.weights_memory_cell = tf.Variable(tf.truncated_normal([1, self.hidden_layer], stddev=0.05))
        self.weights_memory_cell_hidden = tf.Variable(tf.truncated_normal([self.hidden_layer, self.hidden_layer], stddev=0.05))
        self.bias_memory_cell = tf.Variable(tf.zeros([self.hidden_layer]))

        # output layer weigts
        self.weights_output = tf.Variable(tf.truncated_normal([self.hidden_layer, 1], stddev=0.05))
        self.bias_output_layer = tf.Variable(tf.zeros([1]))
