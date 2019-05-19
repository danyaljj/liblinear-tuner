import sys
sys.path.append('/Users/daniel/Desktop/liblinear-2.300/python')
from liblinearutil import *

# trainfile = '/Users/daniel/Desktop/drive-download-20190519T065214Z-001/train.feature'
# devfile = '/Users/daniel/Desktop/drive-download-20190519T065214Z-001/dev.feature'

trainfile = '/Users/daniel/Desktop/propbank/train.feature'
devfile = '/Users/daniel/Desktop/propbank/dev.feature'


print 'reading training data ... '
train_y, train_x = svm_read_problem(trainfile)
print 'reading dev data ... '
dev_y, dev_x = svm_read_problem(devfile)

for i in xrange(-10, 10):
    c = pow(2, i)
    print '\nc=', c
    model = train(train_y, train_x, '-s 5 -c '+str(c)+' -q -B 1 -e 0.001')

    p_label, p_acc, p_val = predict(dev_y, dev_x, model)

