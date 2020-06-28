import nltk
import pickle
from nltk.tokenize import word_tokenize

from nltk.corpus import brown
from nltk.corpus import indian
from mlxtend.evaluate import paired_ttest_kfold_cv
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from nltk import MaxentClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import itertools
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
import sklearn_crfsuite

class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
        
    def split(self, text):
        """
	input format: a paragraph of text
	output format: a list of lists of words.
	e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
	"""
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences
def ngramTagger(train_sents, n=2, defaultTag='NN'):
    t0 = nltk.DefaultTagger(defaultTag)
    if (n <= 0):
        return t0
    elif (n == 1):
        t1 = nltk.UnigramTagger(train_sents, backoff=t0,verbose=True)
        return t1
    elif (n == 2):
        t1 = nltk.UnigramTagger(train_sents, backoff=t0,verbose=True)
        t2 = nltk.BigramTagger(train_sents, backoff=t1,verbose=True)
        return t2
    else:
        t1 = nltk.UnigramTagger(train_sents, backoff=t0,verbose=True)
        t2 = nltk.BigramTagger(train_sents, backoff=t1,verbose=True)
        t3 = nltk.TrigramTagger(train_sents, backoff=t2,verbose=True)
        return t3
def features(sentence, index):
    currWord = sentence[index][0]
    if (index > 0):
        prevWord = sentence[index - 1][0]
    else:
        prevWord = '<START>'
    if (index < len(sentence)-1):
        nextWord = sentence[index + 1][0]
    else:
        nextWord = '<END>'
    return {
        'word': currWord,
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'curr_is_title': currWord.istitle(),
        'prev_is_title': prevWord.istitle(),
        'next_is_title': nextWord.istitle(),
        'curr_is_lower': currWord.islower(),
        'prev_is_lower': prevWord.islower(),
        'next_is_lower': nextWord.islower(),
        'curr_is_upper': currWord.isupper(),
        'prev_is_upper': prevWord.isupper(),
        'next_is_upper': nextWord.isupper(),
        'curr_is_digit': currWord.isdigit(),
        'prev_is_digit': prevWord.isdigit(),
        'next_is_digit': nextWord.isdigit(),
        'curr_prefix-1': currWord[0],
        'curr_prefix-2': currWord[:2],
        'curr_prefix-3': currWord[:3],
        'curr_suffix-1': currWord[-1],
        'curr_suffix-2': currWord[-2:],
        'curr_suffix-3': currWord[-3:],
        'prev_prefix-1': prevWord[0],
        'prev_prefix-2': prevWord[:2],
        'prev_prefix-3': prevWord[:3],
        'prev_suffix-1': prevWord[-1],
        'prev_suffix-2': prevWord[-2:],
        'prev_suffix-3': prevWord[-3:],
        'next_prefix-1': nextWord[0],
        'next_prefix-2': nextWord[:2],
        'next_prefix-3': nextWord[:3],
        'next_suffix-1': nextWord[-1],
        'next_suffix-2': nextWord[-2:],
        'next_suffix-3': nextWord[-3:],
        'prev_word': prevWord,
        'next_word': nextWord,
    }

def transformDatasetSequence(sentences):
    wordFeatures, wordLabels = [], []
    for sent in sentences:
        feats, labels = [], []
        for index in range(len(sent)):
            feats.append(pos_features(sent, index))
            labels.append(sent[index][1])
        wordFeatures.append(feats)
        wordLabels.append(labels)
    return wordFeatures, wordLabels
def pos_features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    features = {
        'word': sentence[index][0],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        #'pos-1': '' if index == 0 else sentence[index-1][1],
        #'pos-2': '' if ((index-1) == 0 or index==0) else sentence[index-2][1],
        #'pos+1': '' if index == len(sentence) - 1 else sentence[index+1][1],
        #'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        #'is_all_caps': sentence[index].upper() == sentence[index],
        #is_all_lower': sentence[index].lower() == sentence[index],
       'prefix-1': sentence[index][0][:1],
       'prefix-2': sentence[index][0][:2],
       'prefix-3': sentence[index][0][:3],
        'suffix-1': sentence[index][0][-1:],
        'suffix-2': sentence[index][0][-2:],
        'suffix-3': sentence[index][0][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1][0],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1][0],
        'has_hyphen': '-' in sentence[index][0],
        'is_numeric': sentence[index][0].isdigit(),
        #'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }
    if index == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[index-1][0]
    return features
def transformDataset(sentences):
    wordFeatures = []
    wordLabels = []
    for sent in sentences:
        for index in range(len(sent)):
            wordFeatures.append(pos_features(sent, index))
            wordLabels.append(sent[index][1])
    return wordFeatures, wordLabels

def trainDecisionTree(trainFeatures, trainLabels):
    clf = make_pipeline(DictVectorizer(sparse=True), OneVsRestClassifier(DecisionTreeClassifier(criterion='entropy')))
    
    #scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    filename = 'DT_hposmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf
def trainRF(trainFeatures,trainLabels):
    clf = make_pipeline(DictVectorizer(sparse=True),RandomForestClassifier())
    clf.fit(trainFeatures,trainLabels)
    filename = 'RF_hposmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf
def trainkNeighbour(trainFeatures, trainLabels):
    print("k clss")
    clf = make_pipeline(DictVectorizer(sparse=True), KNeighborsClassifier(n_neighbors=3))
    #clf = KNeighborsClassifier(n_neighbors=3)
   # scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    filename = 'KNN_hposmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))  
    return clf
def trainNaiveBayes(trainFeatures, trainLabels):
    clf = make_pipeline(DictVectorizer(sparse=True), MultinomialNB())
  #  scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    filename = 'NB_hposmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf
def trainNN(trainFeatures, trainLabels):
    clf = make_pipeline(DictVectorizer(sparse=True),
                        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100), random_state=1))
    #scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    filename = 'NN_hposmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf
def trainMaxentropy(trainFeatures, trainLabels):
    import shorttext
    from shorttext.classifiers import MaxEntClassifier
 
    classifier = MaxEntClassifier()
    clf = make_pipeline(DictVectorizer(sparse=True), MaxentClassifier(encoding=None,weights=0))
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    return clf, scores.mean(),scores
def trainCRF(trainFeatures, trainLabels):
    clf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    )
  #  scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    filename = 'CRF_hposmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print("POS tagger1")
marathi_sent = indian.sents('marathi_pos_rad_3NOV17.pos')
mpos= indian.tagged_sents('marathi_pos_rad_3NOV17.pos')
mp=shuffle(mpos)


print("POS tagger")
size = int(len(marathi_sent) * 0.67)
tags = [tag for (word, tag) in indian.tagged_words('marathi_pos_rad_3NOV17.pos')]
#print(tags)
print(np.unique(tags))
#print("no. of tags=",len(nltk.FreqDist(tags)))
defaultTag = nltk.FreqDist(tags).max()

#print(defaultTag)
train_sents = mp[:size]
#print(len(train_sents))
test_sents = mp[size:]

print(marathi_sent[0])
trainFeatures, trainLabels = transformDataset(train_sents)

testFeatures, testLabels = transformDataset(test_sents)
print("lengths of features")
print(len(trainFeatures), len(trainLabels),len(testFeatures), len(testLabels))
keys={"CC":1, "CCD":2, "CCS":3, "DM":4, "DMD":5, "DMQ":6, "DMR":7, "ECH":8, "INTF":9, "JJ":10, "NEG":11, "NN":12 ,"NNP":13,"NST":14, "PR":15, "PRC":16, "PRF":17,
        "PRL":18, "PRP":19, "PRQ":20, "PUNC":21, "QT":22, "QTC":23, "QTF":24, "QTO":25,"RB":26,"RDF":27, "RP":28,"SYM":29, "UNK":30, "VAUX":31, "VM":32}
tf = testFeatures.copy()
fe = trainFeatures.copy()
features = fe
features.extend(tf)
trl = trainLabels.copy()
labels = trl
tl = testLabels.copy()
labels.extend(tl)
print("lengths of features after copy")
print(len(trainFeatures), len(trainLabels),len(testFeatures), len(testLabels))

print(type(features))

lab = []

for i in range(len(labels)):
    
    if(keys[labels[i]]):
        lab.append(int(keys[labels[i]]))



y = np.int_(lab)
#print("features",fe)
#print(trainFeatures[1])
#print("length of train features ="+str(len(trainFeatures)))
var = 1

while var == 1:
    print("******************MENU********************")
    print("case 1: Unigramtagger")
    print("case 2: Bigramtagger")
    print("case 3: Trigramtagger")
    print("case 4:Naive Bayes classifier")
    print("case 5: Decision tree classifier")
    print("case 6: Neural network")
    print("case 7: K nearest neighbour")
    print("case 8: Conditional Random Fields")
    print("case 9: Random forest classifier")
    print("case 10:exit")   
    print("enter your choice")
    ch=input()
    if ch == "10":
        
        var = 2
        continue
    elif ch == "1":
        tagger = ngramTagger(train_sents, 1 , defaultTag)
        #sent="मी शाळेत जातो."
        #print(tagger.tag(marathi_sent[0]))
        
        print(tagger.evaluate(test_sents))
        continue
    elif ch == "2":
        tagger = ngramTagger(train_sents, 2 , defaultTag)
        print(tagger.evaluate(test_sents))
        continue
    elif ch == "3":
        tagger = ngramTagger(train_sents, 3 , defaultTag)
        print(tagger.evaluate(test_sents))
        continue
    elif ch == "4":
        print("naive bayes")
        tree_model= trainNaiveBayes(trainFeatures, trainLabels)
        NB = tree_model
        
    elif ch =="5":
        tree_model = trainDecisionTree(trainFeatures, trainLabels)
        DT = tree_model
    elif ch == "6":
        tree_model= trainNN(trainFeatures, trainLabels)
        NN = tree_model
    elif ch == "7":
        tree_model = trainkNeighbour(trainFeatures, trainLabels)
        KNN = tree_model
    elif ch == "8":
        trainseqFeatures, trainLabels = transformDatasetSequence(train_sents)
        testseqFeatures, testLabels = transformDatasetSequence(test_sents)
        
        tree_model=trainCRF(trainseqFeatures, trainLabels)
        CRF = tree_model
    elif ch == "9":
        tree_model= trainRF(trainFeatures, trainLabels)
        RF = tree_model
    
    Max = 0
  #  for i in range(1,6):
  #      print("accuracy in fold ",i, " =",scores[i-1])
            
            
   # print("accuracy on train data  = ")
   # print(tree_model_cv_score)
    print("accuracy on test data  = ")
    if(ch == "8"):
        print(tree_model.score(testseqFeatures, testLabels))
    else:
        print(tree_model.score(testFeatures, testLabels))
    if(ch == "8"):
        y_pred = tree_model.fit(trainseqFeatures, trainLabels).predict(testseqFeatures)
    else:
        y_pred = tree_model.fit(trainFeatures, trainLabels).predict(testFeatures)
       #print("y_predicted=",y_pred)    
       # print("y_pred unique= ",np.unique(y_pred))
        #print("y_test unque=",np.unique(testLabels))

#scatter plot
#plt.figure()
#plt.scatter(testLabels, y_pred)
#plt.xlabel("True Values")
#plt.ylabel("Predictions")
#plt.show()
#end
    classes=['CC', 'CCD', 'CCS', 'DM', 'DMD', 'DMQ', 'DMR', 'ECH', 'INTF', 'JJ', 'NEG', 'NN' ,'NNP','NST', 'PR', 'PRC', 'PRF', 'PRL', 'PRP', 'PRQ', 'PUNC', 'QT', 'QTC', 'QTF', 'QTO',
         'RB', 'RDF', 'RP', 'SYM', 'UNK', 'VAUX', 'VM']
    if(ch == "8"):
        labels={"CC":0, "CCD":1, "CCS":2, "DM":3, "DMD":4, "DMQ":5, "DMR":6, "ECH":7, "INTF":8, "JJ":9, "NEG":10, "NN":11 ,"NNP":12,"NST":13, "PR":14, "PRC":15, "PRF":16,
        "PRL":17, "PRP":18, "PRQ":19, "PUNC":20, "QT":21, "QTC":22, "QTF":23, "QTO":24,"RB":25,"RDF":26, "RP":27,"SYM":28, "UNK":29, "VAUX":30, "VM":31}
        # Convert the sequences of tags into a 1-dimensional array
        pred = np.array([tag for row in y_pred for tag in row])
        truth = np.array([tag for row in testLabels for tag in row])
        testLabels=truth
        print(testLabels[0])
        y_pred=pred
        cnf_matrix = confusion_matrix(testLabels, y_pred)
    else:
        cnf_matrix = confusion_matrix(testLabels, y_pred)
    np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.unique(tags),title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
    
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=np.unique(tags), normalize=True,title='Normalized confusion matrix')


    random_state = np.random.RandomState(0)
    
    #if(ch!="8"):
    te_lab = label_binarize(testLabels,classes )
    y_p=label_binarize(y_pred,classes )

#print("len of te_lab",len(te_lab))
#print("len of y_pr",len(y_p))

    precision = dict()
    recall = dict()
    ther = dict()
    average_precision = dict()


    print(classification_report(te_lab, y_p, target_names=classes))
           
    
    #for i in range(0,len(classes)):
    #    precision[i], recall[i], ther[i] = precision_recall_curve(te_lab[:, i],
                                                        #y_p[:, i])
     #   average_precision[i] = average_precision_score(te_lab[:, i], y_p[:, i])
#for i in range(0,len(classes)):
#    print("Precision of",classes[i],"=",precision[i])
#    print("Recall of",classes[i],"=",recall[i])
#    print("Threshold of",classes[i],"=",ther[i])
    
# A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(te_lab.ravel(),y_p.ravel())
    average_precision["micro"] = average_precision_score(te_lab, y_p,average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))


    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,where='post')
    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))

    plt.show()
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range (0,len(classes)):
        fpr[i], tpr[i], _ = roc_curve(te_lab[:, i], y_p[:, i])
        print("fpr len",len(fpr[i]),"tpr len",len(tpr[i]))
        print("fpr",fpr[i])
        print("tpr" ,tpr[i])
        print("te_lab",len(te_lab[:,i]))
        print("y_p",len(y_p[:,i]))
        roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
    for i in range (0,len(classes)):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        tit='Receiver operating characteristic example ='+classes[i]
        plt.title(tit)
        #plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    
    if ch == "4":
        nb_ypred = y_p
    elif ch == "5":
        dt_ypred = y_p
    elif ch == "6":
        nn_ypred = y_p
    elif ch == "7":
        knn_ypred = y_p
    elif ch == "8":
        crf_ypred = y_p
    elif ch == "9":
        rf_ypred = y_p
    elif ch == "10":
        hmm_ypred = y_p
        



print("mcnemar test among NB and DT")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if((np.array_equal(te_lab[i],nb_ypred[i])) and (np.array_equal(te_lab[i],dt_ypred[i]))):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],nb_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],dt_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among NB and DT")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')

print("mcnemar test among NB and NN")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],nb_ypred[i]) and np.array_equal(te_lab[i],nn_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],nb_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],nn_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among NB and NN")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')

print("mcnemar test among NB and KNN")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],nb_ypred[i]) and np.array_equal(te_lab[i],knn_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],nb_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],knn_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among NB and KNN")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')



print("mcnemar test among NB and CRF")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],nb_ypred[i]) and np.array_equal(te_lab[i],crf_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],nb_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],crf_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among NB and CRF")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')



print("mcnemar test among NB and RF")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],nb_ypred[i]) and np.array_equal(te_lab[i],rf_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],nb_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],rf_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among NB and RF")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')






print("mcnemar test among DT and NN")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],dt_ypred[i]) and np.array_equal(te_lab[i],nn_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],dt_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],nn_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among DT and NN")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table,exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')


print("mcnemar test among DT and KNN")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],dt_ypred[i]) and np.array_equal(te_lab[i],knn_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],dt_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],knn_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among DT and KNN")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')

print("mcnemar test among DT and CRF")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],dt_ypred[i]) and np.array_equal(te_lab[i],crf_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],dt_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],crf_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among DT and CRF")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')



print("mcnemar test among DT and RF")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],dt_ypred[i]) and np.array_equal(te_lab[i],rf_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],dt_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],rf_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among DT and RF")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')










print("mcnemar test among NN and KNN")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if((np.array_equal(te_lab[i],y_pred[i])) and (np.array_equal(te_lab[i],knn_ypred[i]))):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],nn_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],knn_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among NN and KNN")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')

print("mcnemar test among NN and CRF")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],nn_ypred[i]) and np.array_equal(te_lab[i],crf_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],nn_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],crf_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among NN and CRF")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')


print("mcnemar test among NN and RF")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],nn_ypred[i]) and np.array_equal(te_lab[i],rf_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],nn_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],rf_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among NN and RF")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')

	

print("mcnemar test among KNN and CRF")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],knn_ypred[i]) and np.array_equal(te_lab[i],crf_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],knn_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],crf_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among KNN and CRF")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')


print("mcnemar test among KNN and RF")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],knn_ypred[i]) and np.array_equal(te_lab[i],rf_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],knn_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],rf_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among KNN and RF")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')



print("mcnemar test among CRF and RF")
#nb_ypred,dt_ypred,nn_ypred,knn_ypred,sv_ypred,te_lab
c1=0
c2=0
c3=0
c4=0
#print(nb_ypred)
for i in range(len(te_lab)):
    if(np.array_equal(te_lab[i],crf_ypred[i]) and np.array_equal(te_lab[i],rf_ypred[i])):
        c1 = c1 + 1
    elif(np.array_equal(te_lab[i],crf_ypred[i])):
        c2 = c2 + 1
    elif(np.array_equal(te_lab[i],rf_ypred[i])):
        c3 = c3 + 1
    else:
        c4 = c4 + 1
print("Contingency table among CRF and RF")
print(c1,"\t" ,c2)
print(c3,"\t",c4)
# define contingency table
table = [[c1,c2],
		 [c3, c4]]
# calculate mcnemar test
result = mcnemar(table, exact=False, correction=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')












