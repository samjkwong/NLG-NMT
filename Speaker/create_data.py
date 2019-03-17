import sys
import csv
import os

csv.field_size_limit(sys.maxsize)
input_file = '../data/the_office/the_office_scripts.csv'
NAME_IND = 5
input_file = '../data/the_office/the_office_scripts.csv'
train_file = 'data/train.txt'
valid_file = 'data/valid.txt'
test_file = 'data/test.txt'

def get_num_lines():
    num_lines = 0
    with open(input_file, encoding='ISO-8859-1') as input:
        reader = csv.reader(input)
        next(reader)
        for row in enumerate(reader):
            num_lines += 1
    return num_lines

def create(num_lines):
    with open(input_file, encoding='ISO-8859-1') as input:
        reader = csv.reader(input)
        next(reader)
        train = open(train_file, 'w')
        valid = open(valid_file, 'w')
        test = open(test_file, 'w')
        for i, row in enumerate(reader):
            if i < 0.8 * num_lines:
                train.write(row[NAME_IND].strip() + '\n')
            elif i < 0.9 * num_lines:
                valid.write(row[NAME_IND].strip() + '\n')
            else:
                test.write(row[NAME_IND].strip() + '\n')
        train.close()
        valid.close()
        test.close()

def main():
    if os.stat(train_file).st_size == 0:
        print('Adding names to data files')
        num_lines = get_num_lines()
        create(num_lines)
    else:
        print('Already added names to data files')

if __name__ == '__main__':
    main()