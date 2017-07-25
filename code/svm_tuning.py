import matplotlib.pyplot as graph_plot
from sklearn import svm
from sys import argv
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_curve , auc
from sklearn.model_selection import KFold
from scipy import interp
import numpy
from itertools import cycle

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
target = numpy.array(target)
train = [x[1:] for x in dataset]
train = numpy.array(train)
output_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#Create Folds
number_of_folds = 10
k_fold = KFold(n_splits = number_of_folds)
#The parameters for each experiment
experiments = [
	[50, 0.1, 1e-10],
	[60, 0.01, 1e-10],
	[50, 1, 1e-8],
	[40, 0.01, 1e-9],
	[40, 0.1, 1e-8],
]
experiment_number = 1
for experiment in experiments:
	print "Experiment: %d " % experiment_number
	#Create a file to save the confusion matrix
	cm_file = open("../Confusion Matrix/SVM/Exp_%d.txt" % experiment_number, 'w')
	#Initialize the performance values
	k = 1
	mean_accuracy = 0
	mean_precision = 0
	mean_recall = 0
	mean_auc = 0
	number_of_svms = experiment[0]
	for train_index, test_index in k_fold.split(train):
		#Get the fold train, train target, test, test target
		fold_train = train[train_index]
		fold_test = train[test_index]
		fold_target_train = target[train_index]
		fold_target_test = target[test_index]
		#Create the classifier model
		svm_bagging_classifier = OneVsRestClassifier(BaggingClassifier(svm.SVC(C = experiment[1], gamma = experiment[2], probability = True), max_samples = 1.0 / number_of_svms, n_estimators = number_of_svms, n_jobs = cpu_count))
		#Fit the classifier model on the train data
		svm_bagging_classifier.fit(fold_train, fold_target_train)
		#Predict the results for test data
		predictions = svm_bagging_classifier.predict(fold_test)
		#Get the probability estimates used for AUROC
		scores = svm_bagging_classifier.predict_proba(fold_test)
		#Binarize the output target as since it is a multi classification model and we get probability estimates for each class available
		binarized_outputs = label_binarize(fold_target_test, classes = output_classes)
		#Calculate the false positive rate and the true positive rate for each of the labels
		false_positive_rate = dict()
		true_positive_rate = dict()
		roc_auc = dict()
		for i in range(len(output_classes)):
			false_positive_rate[i], true_positive_rate[i], _ = roc_curve(binarized_outputs[:, i], scores[:, i])
			roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

		#Calculate the micro rates
		false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(binarized_outputs.ravel(), scores.ravel())
		roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])
		all_false_positive_rate = numpy.unique(numpy.concatenate([false_positive_rate[i] for i in range(len(output_classes))]))

		#Interpolate all ROC curves of the different labels at this points
		mean_true_positive_rate = numpy.zeros_like(all_false_positive_rate)
		for i in range(len(output_classes)):
			mean_true_positive_rate += interp(all_false_positive_rate, false_positive_rate[i], true_positive_rate[i])

		#Average it and compute AUC (average is macro)
		mean_true_positive_rate /= len(output_classes)

		false_positive_rate["macro"] = all_false_positive_rate
		true_positive_rate["macro"] = mean_true_positive_rate
		roc_auc["macro"] = auc(false_positive_rate["macro"], true_positive_rate["macro"])
		lw = 2
		graph_plot.figure()
		#Plot the graph for this fold
		graph_plot.plot(false_positive_rate["micro"], true_positive_rate["micro"],
			label='micro-average ROC curve (area = {0:0.2f})'
			''.format(roc_auc["micro"]),
			color='deeppink', linestyle=':', linewidth=4)

		graph_plot.plot(false_positive_rate["macro"], true_positive_rate["macro"],
			label='macro-average ROC curve (area = {0:0.2f})'
			''.format(roc_auc["macro"]),
			color='navy', linestyle=':', linewidth=4)

		colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'dimgray', 'lemonchiffon', 'mintcream', 'olive', 'slategray', 'tan', 'teal'])
		for i, color in zip(range(len(output_classes)), colors):
			graph_plot.plot(false_positive_rate[i], true_positive_rate[i], color=color, lw=lw,
				label='ROC curve of class {0} (area = {1:0.2f})'
				''.format(i, roc_auc[i]))

		graph_plot.plot([0, 1], [0, 1], 'k--', lw=lw)
		graph_plot.xlim([0.0, 1.0])
		graph_plot.ylim([0.0, 1.05])
		graph_plot.xlabel('False Positive Rate')
		graph_plot.ylabel('True Positive Rate')
		graph_plot.legend(loc="lower right")
		#Save the figure and close the plot
		graph_plot.savefig('../Graphs/SVM/Experiment_%d/roc_%d_fold.png' % (experiment_number, k))
		graph_plot.close('all')
		#Calculate the performance metrics
		accuracy = accuracy_score(fold_target_test, predictions)
		mean_accuracy += accuracy
		precision = precision_score(fold_target_test, predictions, average = 'macro')
		mean_precision += precision
		recall = recall_score(fold_target_test, predictions, average = 'macro')
		mean_recall += recall
		mean_auc += roc_auc['macro']
		print "\tFold %d\n\tAccuracy: %.2f\n\tPrecision: %.2f\n\tRecall: %.2f\n\tAuc: %.2f" % (k, accuracy, precision, recall, roc_auc['macro'])
		fold_cm = confusion_matrix(fold_target_test, predictions)
		cm_file.write('Fold - %d\n%s\n' % (k, fold_cm))
		k += 1
	mean_accuracy = mean_accuracy / number_of_folds
	mean_precision = mean_precision / number_of_folds
	mean_recall = mean_recall / number_of_folds
	mean_auc = mean_auc / number_of_folds
	print "Mean\nAccuracy: %.2f\nPrecision: %.2f\nRecall: %.2f\nAuc: %.2f" % (mean_accuracy, mean_precision, mean_recall, mean_auc)
	experiment_number += 1