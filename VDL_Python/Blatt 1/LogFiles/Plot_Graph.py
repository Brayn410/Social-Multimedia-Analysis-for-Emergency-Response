import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import csv

# Get current working directory cwd:
cur_directory = os.getcwd() + "\\"
log_dir = str(os.path.abspath(os.path.join(cur_directory, os.pardir))) + "\\LogFiles\\"
print("Log Directory =", log_dir)


def read_log_file():
    phases = {"train"}

    train_epochs = list()
    train_loss = list()
    train_acc = list()


    filename = log_dir + "train_log.csv"
    file = open(filename, "r")
    csvReader = csv.reader(file, delimiter='\t')
    header = csvReader.__next__()
    print(header)


    for row in csvReader:
        train_epochs.append(int(row[0]))
        train_loss.append(float("{0:.3f}".format(float(row[1]))))
        train_acc.append(float("{0:.3f}".format(float(row[2]))))

    print(train_epochs)
    print(train_loss)
    print(train_acc)


    plt.figure(num=1, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left = 0.05, right= 0.95, wspace=0.5, )

    # Beginn des plottens:
    plt.style.use('seaborn-whitegrid')
    plt.subplot(121)

    # Plot Axes-Labeling
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plot range of y-axes
    plt.gca().set_ylim([-0.5, train_loss[0]+1])

    # Plot Curve
    plt.plot(train_epochs, train_loss, 'b', label="Train Loss")


    # Plot Legend
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    # plt.show()


    plt.subplot(122)

    # Plot Axes-Labeling# Plot Legend
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Plot range of y-axes
    plt.gca().set_ylim([-0.05, 1.0])

    # Plot Curve
    plt.plot(train_epochs, train_acc, 'b', label="Train Acc")


    # Plot Legend
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


read_log_file()

