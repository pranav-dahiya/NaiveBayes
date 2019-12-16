import glob
import pickle
import numpy as np
from multiprocessing import Process, Lock
from nltk.stem import WordNetLemmatizer


def compute_probability(vocabulary, ignore=0):
    conditional_probability = [{key:1 for key in vocabulary.keys()}, {key:1 for key in vocabulary.keys()}]
    priori_probability = [0, 0]
    lemmatizer = WordNetLemmatizer()
    for i in range(1,11):
        if i != ignore:
            filenames = glob.glob("lingspam/part"+str(i)+"/*.txt")
            for filename in filenames:
                flag = {key:True for key in vocabulary.keys()}
                label = int("spmsg" in filename)
                priori_probability[label] += 1
                with open(filename, "rb") as f:
                    text = f.readlines()
                    for line in text:
                        for word in line.decode().split(" "):
                            word = lemmatizer.lemmatize(word)
                            if word in vocabulary.keys() and flag[word]:
                                conditional_probability[label][word] += 1
                                flag[word] = False
    conditional_probability[0] = {key:value/priori_probability[0] for key, value in conditional_probability[0].items()}
    conditional_probability[1] = {key:value/priori_probability[1] for key, value in conditional_probability[1].items()}
    total = priori_probability[0] + priori_probability[1]
    priori_probability[0] /= total
    priori_probability[1] /= total
    return conditional_probability, priori_probability


def classify(conditional_probability, priori_probability, filename):
    spam, ham = 1, 1
    words = list(conditional_probability[0].keys())
    lemmatizer = WordNetLemmatizer()
    with open(filename, "rb") as f:
        text = f.readlines()
        for line in text:
            for word in line.decode().split(" "):
                word = lemmatizer.lemmatize(word)
                if word in words:
                    ham *= conditional_probability[0][word]
                    spam *= conditional_probability[1][word]
                    words.remove(word)
    for word in words:
        ham *= 1-conditional_probability[0][word]
        spam *= 1-conditional_probability[1][word]
    ham *= priori_probability[0]
    spam *= priori_probability[1]
    if spam > ham:
        return 1
    else:
        return 0


def test(conditional_probability, priori_probability, folder):
    filenames = glob.glob("lingspam/part"+str(folder)+"/*.txt")
    TP, FP, TN, FN = 0, 0, 0, 0
    for filename in filenames:
        label = int("spmsg" in filename)
        label_ = classify(conditional_probability, priori_probability, filename)
        #print(filename, label, label_)
        if label and label_:
            TP += 1
        elif label and not(label_):
            FN += 1
        elif not(label) and not(label_):
            TN += 1
        else:
            FP += 1
    return TP, FP, TN, FN


def validate(vocabulary, i, lock):
    conditional_probability, priori_probability = compute_probability(vocabulary, i)
    TP, FP, TN, FN = test(conditional_probability, priori_probability, i)
    lock.acquire()
    print(i, ",", TP, ",", FP, ",", TN, ",", FN, ",")
    lock.release()


if __name__ == '__main__':
    for v in range(1,5):
        fname = "vocabulary"+str(v)+".pickle"
        print(fname)
        with open(fname, "rb") as f:
            vocabulary = pickle.load(f)
        lock = Lock()
        process_list = [Process(target=validate, args=(vocabulary, i, lock)) for i in range(1,11)]
        for process in process_list:
            process.start()
        for process in process_list:
            process.join()
