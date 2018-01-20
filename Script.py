#music companies (Amazon/Spotify/Pandora)
#import packages
import pandas as pd
import csv
import string
import nltk
import nltk.classify.util
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

#Load training data
inpReviews = csv.reader(open('C:/Users/gaura/OneDrive/Desktop/MIS Project/Trainingset.csv'), delimiter=',')

#function for pre-process
def processReview(Review):
	#Remove Punctuations and make lower case
	translator = str.maketrans('', '', string.punctuation)
	Review = Review.lower()
	Review = Review.translate(translator)
	#Tokenize the Review
	tokens = word_tokenize(Review)
	#check if the words are alpha
	tokens = [word for word in tokens if word.isalpha()]
	stop_words = stopwords.words('english')
	newstopwordslist = []
	#append new stopwords
	addStopWords = open('C:/Users/gaura/OneDrive/Desktop/MIS Project/newstopwords.txt')
	for i in addStopWords:
		newstopwordslist.append(i.rstrip('\n'))
	words = set([w for w in tokens if not w in stop_words and len(w)>2])
	words = set([w for w in words if not w in newstopwordslist and len(w)>2]) #remove additional stopwords words
	return(words)

#function for feature extraction
def extract_feature(Review):
	review_words = Review
	features = {}
	for word in featureList_unique:
		features['contains(%s)' % word] = (word in review_words)
	return(features)

finalReview = []
featureList = []
featureVector = []
featureList_unique = []
for row in inpReviews:
    sentiment = row[0]
    Review = row[1]
    processedReview = processReview(Review)
    featureVector.append(processedReview)
    finalReview.append((processedReview,sentiment))

for i in featureVector:
        for j in i:
                featureList.append(j)
featureList = list(featureList)

#calculate frequency of words
fdist=FreqDist(featureList)
featureList_unique = set(featureList)
#for word, frequency in fdist.most_common(3000):
	#print(u'{};{}'.format(word, frequency))
#print(featureList)
#print(finalReview)
#print(extract_feature(Review))

#Generate the training set
training_set = nltk.classify.util.apply_features(extract_feature, finalReview)
trainset=training_set[1:]

#Train the classifier
#NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

#save pickle file
#save_classifier = open("C:/Users/gaura/OneDrive/Desktop/MIS Project/naivebayes_final.pickle","wb")
#pickle.dump(NBClassifier, save_classifier)
#save_classifier.close()

#load pickle file
classifier_open = open("C:/Users/gaura/OneDrive/Desktop/MIS Project/naivebayes_final.pickle", "rb")
NBClassifier = pickle.load(classifier_open)
classifier_open.close()

#load testing data
testdata = csv.reader(open('C:/Users/gaura/OneDrive/Desktop/MIS Project/testdata.csv'), delimiter=',')
final_output_array = []
for row in testdata:
	date = row[0]
	Review = row[1]
	source = row[2]
	processedTestTweet = processReview(Review)
	label = NBClassifier.classify(extract_feature(processedTestTweet))
	final_output = date+"^"+Review+"^"+label+"^"+source
	#print(final_output)
	final_output_array.append(final_output)

#export to excel
from openpyxl import Workbook
wb = Workbook()
ws = wb.active
for row, i in enumerate(final_output_array):
	column_cell = 'A'
	ws[column_cell+str(row+2)] = str(i)

wb.save("Final_output3.xlsx")

print(nltk.classify.accuracy(NBClassifier,trainset))
#testtweet="With the new update, the app is working smooth and perfectly fine."
#processedTestTweet = processReview(testtweet)
#print(NBClassifier.classify(extract_feature(processedTestTweet)))
#print(NBClassifier.show_most_informative_features(10))