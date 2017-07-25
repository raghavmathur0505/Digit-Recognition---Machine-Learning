from sys import argv
from sklearn.ensemble import RandomForestClassifier
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
number_of_svms = 50
random_forest_classifier = RandomForestClassifier(n_estimators = 100, max_features = 28, n_jobs = cpu_count)
random_forest_classifier.fit(train, target)
predictions = random_forest_classifier.predict(test)
numpy.savetxt('../Predictions/random_forest_predictions.csv', numpy.c_[range(1,len(test)+1), predictions], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')