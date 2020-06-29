from __future__ import division #To avoid integer division
from operator import itemgetter
from nltk.corpus import indian
import numpy as np
import itertools
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from statistics import mean
from sklearn.metrics import roc_curve, auc
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

from nltk.tokenize import word_tokenize
import time
start_time = time.time()

def fn_train(train_sent):

###Training Phase###
    
    print(len(train_sent))
    marathi_news_tagged = train_sent
    
    num_words_train = len(marathi_news_tagged)
    train_li_words = ['']
    train_li_words*= num_words_train

    train_li_tags = ['']
    train_li_tags*= num_words_train

    for i in range(num_words_train):
        train_li_words[i] = marathi_news_tagged[i][0]
        train_li_tags[i] = marathi_news_tagged[i][1]


    dict2_tag_follow_tag_ = {}
    

    dict2_word_tag = {}
   

    dict_word_tag_baseline = {}
    #Dictionary with word as key and its most frequent tag as value

    for i in range(num_words_train-1):
        outer_key = train_li_tags[i]
        inner_key = train_li_tags[i+1]
        dict2_tag_follow_tag_[outer_key]=dict2_tag_follow_tag_.get(outer_key,{})
        dict2_tag_follow_tag_[outer_key][inner_key] = dict2_tag_follow_tag_[outer_key].get(inner_key,0)
        dict2_tag_follow_tag_[outer_key][inner_key]+=1

        outer_key = train_li_words[i]
        inner_key = train_li_tags[i]
        dict2_word_tag[outer_key]=dict2_word_tag.get(outer_key,{})
        dict2_word_tag[outer_key][inner_key] = dict2_word_tag[outer_key].get(inner_key,0)
        dict2_word_tag[outer_key][inner_key]+=1


    """The 1st token is indicated by being the 1st word of a senetence, that is the word after period(.)
    Adjusting for the fact that the first word of the document is not accounted for that way
    """

    dict2_tag_follow_tag_['.'] = dict2_tag_follow_tag_.get('.',{})
    dict2_tag_follow_tag_['.'][train_li_tags[0]] = dict2_tag_follow_tag_['.'].get(train_li_tags[0],0)
    dict2_tag_follow_tag_['.'][train_li_tags[0]]+=1


    last_index = num_words_train-1

    #Accounting for the last word-tag pair
    outer_key = train_li_words[last_index]
    inner_key = train_li_tags[last_index]
    dict2_word_tag[outer_key]=dict2_word_tag.get(outer_key,{})
    dict2_word_tag[outer_key][inner_key] = dict2_word_tag[outer_key].get(inner_key,0)
    dict2_word_tag[outer_key][inner_key]+=1


    """Converting counts to probabilities in the two nested dictionaries
    & also converting the nested dictionaries to outer dictionary with inner sorted lists
    """
    for key in dict2_tag_follow_tag_:
        di = dict2_tag_follow_tag_[key]
        s = sum(di.values())
        for innkey in di:
            di[innkey] /= s
        di = di.items()
        di = sorted(di,key=lambda x: x[0])
        dict2_tag_follow_tag_[key] = di

    for key in dict2_word_tag:
        di = dict2_word_tag[key]
        dict_word_tag_baseline[key] = max(di, key=di.get)
        s = sum(di.values())
        for innkey in di:
            di[innkey] /= s
        di = di.items()
        di = sorted(di,key=lambda x: x[0])
        dict2_word_tag[key] = di

    return dict2_tag_follow_tag_, dict2_word_tag, dict_word_tag_baseline


###Testing Phase###

def fn_assign_POS_tags(words_list):

    num_words_test = len(words_list)

    output_li = ['']
    output_li*= num_words_test
    for i in range(num_words_test):
        if i==0:    #Accounting for the 1st word in the test document for the Viterbi
            di_transition_probs = dict2_tag_follow_tag_['.']
        else:
            di_transition_probs = dict2_tag_follow_tag_[output_li[i-1]]
            
        di_emission_probs = dict2_word_tag.get(words_list[i],'')

        #If unknown word  - tag = 'NP'
        if di_emission_probs=='':
            output_li[i]='NN'
            
        else:
            max_prod_prob = 0
            counter_trans = 0
            counter_emis =0
            prod_prob = 0
            while counter_trans < len(di_transition_probs) and counter_emis < len(di_emission_probs):
                tag_tr = di_transition_probs[counter_trans][0]
                tag_em = di_emission_probs[counter_emis][0]
                if tag_tr < tag_em:
                    counter_trans+=1
                elif tag_tr > tag_em:
                    counter_emis+=1
                else:
                    prod_prob = di_transition_probs[counter_trans][1] * di_emission_probs[counter_emis][1]
                    if prod_prob > max_prod_prob:
                        max_prod_prob = prod_prob
                        output_li[i] = tag_tr
                        #print "i=",i," and output=",output_li[i]
                    counter_trans+=1
                    counter_emis+=1    
        

        if output_li[i]=='': #In case there are no matching entries between the transition tags and emission tags, we choose the most frequent emission tag
            output_li[i] = max(di_emission_probs,key=itemgetter(1))[0]  
            
    return output_li

#tup = fn_train()
#dict2_tag_follow_tag_ = tup[0]
#dict2_word_tag = tup[1]
#dict_word_tag_baseline = tup[2]

if __name__ == "__main__":

    
    k=5
    #to shuffle sentences
    mp = indian.tagged_words('marathi_pos_rad_3NOV17.pos')
    marathi_sent=shuffle(mp)
    print("length of tagged words=",len(marathi_sent))
    size = int(len(marathi_sent) * 0.8)
    print("size=",size)
    mtrain1=marathi_sent[:size]
    print("len of mtrain=",len(mtrain1))
    test=marathi_sent[size:]
    print("len of mtrain=",len(test))
    # without shufle
    #marathi_sent= indian.tagged_words('marathi_pos_rad_3NOV17.pos')
    r = len(mtrain1)/k
    l=len(mtrain1)
    score=[]
    for i in range(k):
        test_set = mtrain1[int(r*i):int(r*i+r)]
        if i != 0:
            train_set = mtrain1[:int(r*i)]
        else:

            train_set = []
        train_set += mtrain1[int(r*i+r):l]
        print("length of train_set i=",i,"=",len(train_set))
        print("length of test_set=",i,"=",len(test_set))
        tup = fn_train(train_set)
        dict2_tag_follow_tag_ = tup[0]
        dict2_word_tag = tup[1]
        dict_word_tag_baseline = tup[2]
      
        marathi_test_tagged=test_set  
        print("first word",marathi_test_tagged[0][0])
        num_words_test = len(marathi_test_tagged)

        test_li_words = ['']
        test_li_words*= num_words_test

        test_li_tags = ['']
        test_li_tags*= num_words_test

        num_errors = 0
        num_errors_baseline = 0
        acc=0
        for j in range(num_words_test):
            temp_li = marathi_test_tagged[j]
            test_li_words[j] = temp_li[0]
            test_li_tags[j] = temp_li[1]


        output_li = fn_assign_POS_tags(test_li_words)
    
        for j in range(num_words_test):
            if output_li[j]!=test_li_tags[j]:
                num_errors+=1
            else:
                acc+=1

        print("Fraction of errors (Viterbi) in fold :",i,"=",(num_errors/num_words_test))
        accuracy=(acc/num_words_test)
        print("Accuracy in fold",i,"= :",accuracy)
        score.append(accuracy)


        print(time.time() - start_time, "seconds")
    print("average accuracy=",mean(score))
    marathi_gov_tagged=test
    print("first word",marathi_test_tagged[0][0])
    num_words_test = len(marathi_test_tagged)

    test_li_words = ['']
    test_li_words*= num_words_test

    test_li_tags = ['']
    test_li_tags*= num_words_test

    num_errors = 0
    num_errors_baseline = 0
    acc=0
    for j in range(num_words_test):
        temp_li = marathi_test_tagged[j]
        test_li_words[j] = temp_li[0]
        test_li_tags[j] = temp_li[1]


    output_li = fn_assign_POS_tags(test_li_words)
    
    for j in range(num_words_test):
        if output_li[j]!=test_li_tags[j]:
            num_errors+=1

        else:
            acc+=1

    print("Fraction of errors (Viterbi) in fold :",i,"=",(num_errors/num_words_test))
    print("Accuracy on test data set= :",(acc/num_words_test))
    classes=['CC', 'CCD', 'CCS', 'DM', 'DMD', 'DMQ', 'DMR', 'ECH', 'INTF', 'JJ', 'NEG', 'NN' ,'NNP','NST', 'PR', 'PRC', 'PRF', 'PRL', 'PRP', 'PRQ', 'PUNC', 'QT', 'QTC', 'QTF', 'QTO',
                'RB', 'RDF', 'RP', 'SYM', 'UNK', 'VAUX', 'VM']
    cnf_matrix = confusion_matrix(test_li_tags, output_li)
    np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes,title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes, normalize=True,title='Normalized confusion matrix')
    plt.show()

    te_lab = label_binarize(test_li_tags,classes )
    y_p=label_binarize(output_li,classes )

#print("len of te_lab",len(te_lab))
#print("len of y_pr",len(y_p))

    precision = dict()
    recall = dict()
    ther = dict()
    average_precision = dict()


    print(classification_report(te_lab, y_p, target_names=classes))
           

    for j in range(0,len(classes)):
        precision[j], recall[j], ther[j] = precision_recall_curve(te_lab[:, j],
                                                        y_p[:, j])
        average_precision[j] = average_precision_score(te_lab[:, j], y_p[:, j])
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

    
