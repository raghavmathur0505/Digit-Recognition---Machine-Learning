from sklearn import svm
from sys import argv
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy

cpu_count = 1
if(len(argv) == 2):
	script, cpu_count = argv
try:
	cpu_count = int(cpu_count)
except Exception, e:
	print "Cpu count should be a number"
	exit()
dataset = numpy.genfromtxt(open('../Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
target = [x[0] for x in dataset]
train = [x[1:] for x in dataset]
test = numpy.genfromtxt(open('../Data/test.csv','r'), delimiter=',', dtype='f8')[1:]
number_of_svms = 40
svm_bagging_classifier = OneVsRestClassifier(BaggingClassifier(svm.SVC(C = 0.01, gamma = 1e-8), max_samples = 1.0 / number_of_svms, n_estimators = number_of_svms, n_jobs = cpu_count))
svm_bagging_classifier.fit(train, target)
predictions = svm_bagging_classifier.predict(test)
numpy.savetxt('../Predictions/svm_predictions.csv', numpy.c_[range(1,len(test)+1), predictions], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')