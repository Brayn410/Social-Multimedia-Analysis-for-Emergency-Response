# In this exercise, we will import libraries when needed so that we understand the need for it.
# However, this is a bad practice and don't get used to it.
import numpy as np
import os

print(os.getcwd())

# read data from reviews and labels file.
with open('data/reviews.txt', 'r') as f:
    reviews_ = f.readlines()
with open('data/labels.txt', 'r') as f:
    labels = f.readlines()

def show_data(data):
    # One of the most important task is to visualize data before starting with any ML task.
    for i in range(5):
        print(labels[i] + "\t: " + data[i])

show_data(reviews_)

# Make everything lower case to make the whole dataset even.
reviews = ''.join(reviews_).lower()



# complete the function below to remove punctuations and save it in no_punct_text

def text_without_punct(reviews):
    exclude = set(string.punctuation)
    # reviews = [ch for ch in reviews if ch not in exclude]
    cleanded = []
    temp_row = []
    for elem in reviews:
        if elem == '\n':
            cleanded.append(''.join(temp_row))
            temp_row = []
        else:
            if not (elem in exclude):
                temp_row.extend(elem)


    return cleanded

no_punct_text = text_without_punct(reviews)
show_data(no_punct_text)

print("\n\n\n", no_punct_text[0])
print("\n\n\n", no_punct_text[0].split(' '))


# split the formatted no_punct_text into words
def split_in_words(no_punct_text):
    erg = [elem.split(' ') for elem in no_punct_text]

    no_empty_words = []
    for row in erg:
        temp = []
        for elem in row:
            if elem != '':
                temp.append(elem)
        no_empty_words.append(temp)

    return no_empty_words

words = split_in_words(no_punct_text)
print("The first ten words in the first sentence:", words[0][:10])
