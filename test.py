import pickle

with open("vocabulary3.txt", "rb") as f:
    lemmatized = pickle.load(f)

with open("vocabulary4.txt", "rb") as f:
    threshold = pickle.load(f)

print(len(threshold))
