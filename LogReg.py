#----------------------------------------
# CS 5033 - SL - Short Project
# Nathan Huffman
# Trains and evaluates logistic regression
#----------------------------------------

#---Parameters---------------------------
# Test name
name = 'Regression'
# Path to data file
directory = 'data'
filename = 'data_banknote_authentication'
# Labels for already prepared data
labels = ('train','validate','test')
# Indicate column with labels
label_colm = -1
#----------------------------------------

import BN_NN as prep
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def eval(model, data, export=False):
    predicts = model.predict(data[0])
    if export: prep.export(predicts, directory+filename+'_'+name)
    return accuracy_score(data[1], predicts)

def train(data):
    classifier = LogisticRegression()
    classifier.fit(data[0], data[1])
    return classifier

# Main driving function
def main():
    print('Method   : Logistic Regression\n-------------------------')
    dataset = prep.DataLoader(directory, filename)
    policy = train(dataset.data[0])
    print('Accuracy    : {:.6f}'.format(eval(policy , dataset.data[1], False)))
    
if __name__ == "__main__":
    main()