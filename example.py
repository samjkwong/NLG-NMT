import csv
import sys

csv.field_size_limit(sys.maxsize)

episode = 2
line_text = 4
speaker = 5
d = {}

name = "the_office_scripts.csv"
train_file = "train.txt"
test_file = "test.txt"
valid_file = "valid.txt"

f = open(name)
train = open(train_file, "w")
test = open(test_file, "w")
valid = open(valid_file, "w") 

reader = csv.reader(f)
next(reader)

for row in reader:
    if int(row[episode]) <= 15:
        train.write(row[speaker] + ": " + row[line_text] + "\n")
    elif int(row[episode]) <= 19:
        test.write(row[speaker] + ": " + row[line_text] + "\n")
    else:
        valid.write(row[speaker] + ": " + row[line_text] + "\n")