import sys
sys.path.append('/Users/daniel/ideaProjects/liblinear-220/python')
from liblinearutil import *


def evaluate(dec_vals, golds):

    # group instances by question
    questions = []
    for i in xrange(len(golds)):
        if golds[i] == 0 and (i == 0 or golds[i-1] == 1):
            questions.append([[golds[i], dec_vals[i]]]) # a new question
        else:
            questions[-1].append([golds[i], dec_vals[i]])

    correct = 0
    for question in questions:
        question.sort(key = lambda x: x[1], reverse=True)
        if question[0][0] == 1: # if top prediction has gold 1
            correct += 1

    print '% of correctly answered questions', "{0:.2f}".format(100*float(correct)/len(questions)), correct, len(questions)


trainfile = '../relational-all-the-negatives/training_size18694_Wed_Mar_14_23:07:25_UTC_2018log.txt'
devfile = '../relational-all-the-negatives/trainheldout_size202_Wed_Mar_14_23:07:25_UTC_2018log.txt'

print 'reading training data ... '
train_y, train_x = svm_read_problem(trainfile)
print 'reading dev data ... '
dev_y, dev_x = svm_read_problem(devfile)

for i in xrange(-10, 10):
    c = pow(2, i)
    print '\nc=', c
    model = train(train_y, train_x, '-s 5 -c '+str(c)+' -q -B 1 -e 0.001')

    p_label, p_acc, p_val = predict(dev_y, dev_x, model)
    p_val = map(lambda x: x[0], p_val)
    if model.label[0] == 0:
        p_val = map(lambda x: -x, p_val)
    evaluate(p_val, dev_y)
