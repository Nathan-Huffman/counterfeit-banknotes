'''
BankNotes_NeuralNetworks
author: Nathan Huffman
version: 0.1

This library implements Deep Neural Net classifiers,
specifically for counterfeit bank note detection.
It also contains wrappers for running this network on any given dataset.
'''

#---Data Parameters------------
data_dir = 'data'
data_name = 'data_banknote_authentication'
label_colm = -1
random = 12345
labels=('train','test','validate')
splits=(0.01,0.79,0.2)
del_columns=()
str_columns=()
str_replace={}
#---Agent Parameters-----------
agent_name = 'NetClassifier'
params = {
'epochs' : 500,
'learning_rate' : 0.01,
'layers' : (2,)}
#---Output Parameters----------
save_dir = 'graphs'
#------------------------------

# Allow CPU to be preferred with '-c' or '--cpu' command line arg
from sys import argv
from os import path, makedirs, environ
if '-c' in argv or '--cpu' in argv:
    environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import required modules (tensorflow must be after CUDA config above)
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import GradientTape, squeeze, convert_to_tensor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions

import numpy as np
from csv import reader, writer
from random import seed, shuffle 

# Formats data for learning data from given csv file
# Includes deleting columns, reformatting string columns, and partitioning data
class DataLoader():
    def __init__(self, directory, filename, label_colm=-1, random=random, labels=labels, splits=splits,
                 del_columns=del_columns, str_columns=str_columns, str_replace=str_replace):
        # Construct the full path to file
        self.raw_path = path.join(directory, filename)
        self.shuffled_name = filename + '_{}'.format(splits) + ('_{}'.format(random) if random else '')
        self.full_path = path.join(directory, self.shuffled_name)
        self.labels = labels
        
        # Either prepare the file or reload it
        if path.isfile(self.full_path+'_'+labels[0]+'.csv'):
            self.data = self.import_all()
        else:
            self.data = self.prep(filename, random, splits, del_columns, str_columns, str_replace)
            self.export_all()
        
        # Split the loaded data via the label colmn
        self.split_all(label_colm)
    
    # Params: ("data_banknote_authentication", (0.6,0.2,0.2), 12345, )
    def prep(self, filename, random=None, splits=(0.6,0.2,0.2), del_columns=(), str_columns=(), str_replace={}):
        partitions = [0] * len(splits)
        with open(self.raw_path+'.csv') as input:
            data = list(reader(input, delimiter=','))[1:]       # Import data into 2d list
            
            for line in data:                                   # Iterate through lines of data for processing
                for x in range(len(str_columns)):
                    target = str_columns[x]
                    line[target] = str_replace[x][line[target]] # Update string data from dictionary
                for x in sorted(list(del_columns),reverse=True):
                    del line[x]                                 # Delete column element from each line
            data = [[float(x) for x in line] for line in data]  # Convert everything to floats

            if random:
                seed(random)                                    # Set seed for repeatable random shuffle
                shuffle(data)                                   # Shuffle data, mixing data in each partition
            
            progress = 0
            lines = sum(1 for line in data)                     # Count number of lines in file
            for i in range(len(splits)):
                size = round(splits[i]*lines)                   # Calculate the size of each partition
                part = data[progress:progress+size]             # Mask to only look at next portion of data, based on size
                partitions[i] = np.array(part)                  # Copy mask of data to spot in partion
                progress += size                                # Keep track of current partition head
            
        return partitions                                       # Return preppred data as numpy array

    # Imports already prepared data
    # Including option to split data from labels
    # Params: ('../Data/Electric_Grid/', 'Data_for_UCI_named', -1)
    def import_data(self, suffix):
        with open(self.full_path+'_'+suffix+'.csv') as input:
            data = list(reader(input, delimiter=','))                       # Import data into 2d list
            data = np.array([[float(x) for x in line] for line in data])    # Convert everything to floats
            return data

    # Bulk imports already prepared data
    # Including option to split data from labels
    # Params: ('../Data/Electric_Grid/', 'Data_for_UCI_named', ('train','vaildate','test'), -1)
    def import_all(self):
        bulk = [0]*len(self.labels)
        for index in range(len(bulk)):
            bulk[index] = self.import_data(self.labels[index])
        return bulk

    # Wraps splitting the data from labels
    @staticmethod
    def split(data, label_colm=-1):
        if label_colm < 0: label_colm = len(data[0]) + label_colm
        data_colms = [i for i in range(len(data[0])) if i != label_colm]
        return data[:, data_colms], data[:, label_colm]

    # Splits all parts of the dataset via the label colm
    def split_all(self, label_colm=-1):
        for i in range(len(self.data)):
            self.data[i] = self.split(self.data[i], label_colm)

    # Merges two datasets, joins two 2d tuples of np arrays
    @staticmethod
    def join(data1, data2):
        merged = [0]*len(data1)
        for index in range(len(data1)):
            merged[index] = np.concatenate((data1[index],data2[index]))
        return merged

    # Exports one file to csv
    @staticmethod
    def export(data, filename):
        with open(filename+'.csv','w+', newline='') as output:
            csv_writer = writer(output, delimiter=',', )    # Use a csv writer to print each line into the file
            [csv_writer.writerow(line) for line in data]

    # Export preppred data to separate files, using labels for name suffixes
    def export_all(self):
        for i in range(len(self.data)):                     # Iterate through each partition to print to rile
            self.export(self.data[i], self.full_path+'_'+self.labels[i])

# Reinforcement learning agent that utilizing Policy Gradient
class NetClassifier:
    def __init__(self, n_inputs, params=params):
        self.total_epochs = 0               # Store total amount of training
        self.n_inputs = n_inputs            # Store dimension of input
        
        self.build_model(params['layers'], params['learning_rate'])
        
    # Create neural network based on layer and learning rate specifications
    def build_model(self, layers, lr):
        state = Input(shape=(self.n_inputs,))           # Create inputs to take in state info 
        build = state
        for layer in layers:                            # Build specified dense hidden layers
            build = Dense(layer, activation='relu')(build)
        probs = Dense(1, activation='sigmoid')(build)   # Add classification layer, with distributed action probabilites
        
        self.model = Model(inputs=[state], outputs=[probs])                                         # Build model with layers
        self.model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy']) # Compile with Adam and classifiers
    
    # Improve own policy, using gradient descent
    def learn(self, train, validate, epochs, early_stop=False):
        # If early stopping, then add appropriate callback
        callbacks = None if not early_stop else EarlyStopping(monitor='loss', patience='5')
        # Fit model, returning history
        return self.model.fit(train[0], train[1], validation_data=(validate[0],validate[1]),
                              epochs=epochs, callbacks=callbacks)
    
    # Evaluate model 
    def eval(self, data):
        for index in range(len(data[0])):
            point = convert_to_tensor(data[0][index].reshape(-1,4))
            answer = convert_to_tensor(data[1][index].reshape(-1,1))
            print('{:50} = {:3} -> {} / {}'.format('{}'.format(data[0][index]), data[1][index], self.model(point), self.model.evaluate(point, answer)))
       
# Encapsulates running of a specified agent on a dataset
class DataRunner:
    def __init__(self, data_dir=data_dir, data_name=data_name, label_colm=label_colm,
                 random=random, agent_name=agent_name, params=params, labels=labels, splits=splits):
        # Save the private fields
        self.episodes_trained = 0
        # Import dataset with default params
        self.dataset = DataLoader(data_dir, data_name, label_colm, random, labels, splits)
        
        # Model parameters
        self.params = params
        # Create model for data
        self.agent = agent_name(n_inputs=len(self.dataset.data[0][0][0]), params=params)
        
    def train(self, epochs=params['epochs'], early_stop=False):    
        # Pass along flag of whether or not to do early stopping
        self.history = self.agent.learn(self.dataset.data[0], self.dataset.data[1], epochs, early_stop)
        
    def eval(self, save_dir=None, debug=False):
        # Eval agent on specified dataset
        if debug:
            self.agent.eval(self.dataset.data[0])
        # Graph training history if save directory spcefied
        if save_dir:
            save_name = '{}_{}.png'.format(self.dataset.shuffled_name, self.params)
            save_loc = path.join(save_dir, save_name).replace(':','').replace(' ','')
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['accuracy'], label='Training Accuracy')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch',fontsize=14); plt.ylabel('Loss / Accuracy',fontsize=14)
            plt.title('Neural Network - Model Learning',fontsize=18)
            plt.legend(loc='center right')
            plt.savefig(save_loc)
        
# Handle optional command line arguemnts and update variables
def handle_args():
    description = "Net Classifier Library | Learns Counterfeit Detection"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c","--cpu", help="Force to execute on CPU", action="store_true")
    parser.add_argument("-e","--epochs", help="Specify num training epochs", type=int)
    parser.add_argument("-a","-lr", "--learning_rate", help="Specify alpha / learning rate", type=float)
    
    args = parser.parse_args()
    
    if args.epochs: global params; params['epochs'] = args.epochs
    if args.learning_rate: global params; params['learning_rate'] = args.learning_rate

    return args

# Main driver, learns Lunar Lander then evaualtes
if __name__ == '__main__':
    args = handle_args()
    
    runner = DataRunner(data_dir, data_name, label_colm, random, NetClassifier, params)
    runner.train()
    runner.eval(save_dir)