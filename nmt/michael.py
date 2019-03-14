import csv
import sys
import json

csv.field_size_limit(sys.maxsize)

episode = 2
line_text = 4
speaker = 5
d = {}

name = "en_es_data/the_office_scripts.csv"
train_speaker = open("en_es_data/train.es", "w")
test_speaker = open("en_es_data/test.es", "w")
dev_speaker = open("en_es_data/dev.es", "w")
train_michael = open("en_es_data/train.en", "w")
test_michael = open("en_es_data/test.en", "w")
dev_michael = open("en_es_data/dev.en", "w")

f = open(name)

reader = csv.reader(f)
next(reader)
prevRow = next(reader)

speaker_vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
michael_vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
data = {"src_word2id": speaker_vocab, "tgt_word2id": michael_vocab}
speaker_count = 1
michael_count = 1

for currRow in reader:
    if currRow[speaker] == "Michael" and prevRow[episode] == currRow[episode]:
        if int(currRow[episode]) <= 15:
            train_speaker.write(prevRow[line_text] + "\n")
            train_michael.write(currRow[line_text] + "\n")
        elif int(currRow[episode]) <= 19:
            test_speaker.write(prevRow[line_text] + "\n")
            test_michael.write(currRow[line_text] + "\n")
        else:
            dev_speaker.write(prevRow[line_text] + "\n")
            dev_michael.write(currRow[line_text] + "\n")
    for word in prevRow[line_text].split(" "):
        if word not in speaker_vocab.keys():
            speaker_vocab[word] = speaker_count
            speaker_count += 1
    for word in currRow[line_text].split(" "):
        if word not in michael_vocab.keys():
            michael_vocab[word] = michael_count
            michael_count += 1
    prevRow = currRow

write_file = open("vocab.json", "w")
json.dump(data, write_file)