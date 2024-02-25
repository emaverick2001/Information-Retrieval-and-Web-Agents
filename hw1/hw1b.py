import argparse
import string
import re
from itertools import groupby

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# TODO question about Segement tag, how do we make our features reprsent that a certain segment tag is present
# TODO do we need to classift #BLANK# tags?
# TODO I dont have a tag for ADDRESS, how do I classify it?
# TODO can we use the re package
class SegmentClassifier:
    # define a constructor that takes in the format type
    def __init__(self, format):
        self.format = format

    def train(self, trainX, trainY): # trainX is a list of segment text, trainY is a list of labels
        self.clf = DecisionTreeClassifier()  # TODO: experiment with different models
        X = [self.extract_features(x) for x in trainX] # for each segment, extract features
        self.clf.fit(X, trainY) # train the model with the features and labels
    
    # text does not include the label
    def extract_features(self, text):
        words = text.split() 
        # print(text)
        # print(words) # words is a list of words in the segment if format is segment else it is a list of words in a line of the segment if format is line
        
        def do_most_lines_start_with_quote_char(text, common_QT_indicators):
            lines = text.split('\n')
            matching_lines = sum(any(word[0] in common_QT_indicators for word in line.split()) for line in lines if line.strip())

            # Check if more than 75% of the lines start with a character in the list
            if matching_lines / len(lines) > 0.75:
                return 1

            return 0

        def do_most_lines_start_with_quote_string(text, common_QT_indicators):
            lines = text.split('\n')
            matching_lines = sum(any(line.strip().startswith(indicator) for indicator in common_QT_indicators) for line in lines if line.strip())

            # Check if more than 75% of the lines start with a string in the list
            if matching_lines / len(lines) > 0.75:
                return 1

            return 0
        
        def is_single_line(text):
            lines = text.split('\n')

            # Remove empty lines
            lines = [line for line in lines if line.strip()]

            if len(lines) == 1:
                return 1

            return 0
        
        def is_mostly_internal_whitespace(text):
            lines = text.split('\n')

            # Remove empty lines
            lines = [line for line in lines if line.strip()]

            # Count lines with internal white space (spaces or tabs)
            whitespace_lines = sum((' ' in line.strip() or '\t' in line.strip()) for line in lines)

            # Check if more than 75% of the lines contain internal white space
            if whitespace_lines / len(lines) > 0.8:
                return 1

            return 0
                
        def is_tabular(text):
            lines = text.split('\n')

            # Check for multiple consecutive spaces or tabs in each line
            for line in lines:
                if '  ' in line or '\t' in line:
                    return True

            return False
        
        def is_mostly_sentences(text):
            lines = text.split('\n')

            # Remove empty lines
            lines = [line for line in lines if line.strip()]

            # Count lines that end with punctuation
            sentence_lines = sum(line.strip()[-1] in string.punctuation for line in lines)

            # Check if more than 75% of the lines are sentences
            if sentence_lines / len(lines) > 0.50:
                return 1

            return 0   
          
        def is_all_caps(text):
            lines = text.split('\n')
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.isupper():
                    return 1
            return 0
        
        def is_line_length_similar(text):
            lines = text.split('\n')
            last_line_length = None
            
            for line in lines[1:]:  # Start from the second line
                # Check if the length of the line is similar to the length of the previous line
                if last_line_length is not None:
                    if abs(len(line) - last_line_length) > 5:
                        return 0

                last_line_length = len(line)

            return 1
        
        def is_mostly_alphanumeric(text):
            lines = text.split('\n')
            
            for line in lines:
                # Count the number of alphanumeric and non-alphanumeric characters in the line
                alphanumeric_count = sum(1 for char in line if char.isalnum())
                non_alphanumeric_count = len(line) - alphanumeric_count

                # Check if the counts are similar
                if abs(alphanumeric_count - non_alphanumeric_count) > 8:
                    return 0

            return 1
        
        def is_mostly_whitespace(text):
            lines = text.split('\n')
            
            for line in lines:
                # Count the number of whitespace and non-whitespace characters in the line
                whitespace_count = line.count(' ') + line.count('\t')
                non_whitespace_count = len(line) - whitespace_count

                # Check if the counts are similar
                if non_whitespace_count / len(line) > 0.5:
                    return 0

            return 1
        
        def is_mostly_non_alpanum(text):
            lines = text.split('\n')
            
            for line in lines:
                # Count the number of non-alphanumeric characters in the line
                non_alnum_count = sum(not char.isalnum() and not char.isspace() for char in line)

                # Check if the count is high
                if non_alnum_count > len(line) / 2:
                    return 1

            return 0
        
        # def is_address(text):
        #     lines = text.split('\n')
        #     for line in lines:
        #         if any(indicator in line for indicator in common_Address_indicators):
        #             return 1
        #     return 0
        
        def contains_common_address_indicator(text, common_Address_indicators):
            # Split the text into words and convert them to lowercase
            words = text.lower().split()

            # Check if any of the words in the text are in the common address indicators list
            if any(word in common_Address_indicators for word in words):
                return 1

            return 0
        
        def starts_with_digit(text):
            lines = text.split('\n')
            for line in lines:
                words = line.split()
                if words[0].strip('().)').isdigit():
                    return 1
            return 0
        
        def starts_with_digit_and_special_char(text):
            lines = text.split('\n')
            for line in lines:
                words = line.split()
                if words[0].endswith('.') or words[0].endswith(')'):
                    if words[0].strip('.)').isdigit():
                        return 1
            return 0
        
        common_NNHEAD_words = ['From', 'Subject', 'Date', 'To', 'Path', 'Cc', 'Bcc', 'Reply-To', 'Return-Path', 'Received', 'Message-ID', 'In-Reply-To', 'References', 'Content-Type', 'Content-Transfer-Encoding', 'MIME-Version', 'X-Mailer', 'X-MSMail-Priority', 'X-Priority', 'X-MSMail-Priority', 'X-Newsreader']
        common_QT_indicators = ['>', '>>', '>>>', ":", "CJK> ","@","KC>",'GDG>','RNA>']
        common_SIG_indicators = ['--', '---', '----','==']
        common_Address_indicators = ['email:', 'phone:', 'fax:', 'address:', 'tel:', 'e-mail:', 'phone', 'fax', 'address', 'tel', 'e-mail']
        common_items_indicators = ['1.', '-','1)','(1)']
        if self.format == 'segment':
            features = [
            len(text),
            len(text.strip()),
            len(words),
            1 if words[0].isupper() and words[0] in common_NNHEAD_words else 0, # check for nnhead
            1 if words[0].isupper() and words[0].endswith(':') else 0, # check for nnhead
            1 if do_most_lines_start_with_quote_char(text, common_QT_indicators) or do_most_lines_start_with_quote_string(text, common_QT_indicators) or words[0] in common_QT_indicators else 0, # check for qt
            1 if text.startswith('In article') or "wrote:" in text else 0, # check for qt
            1 if words[0] in common_SIG_indicators or is_single_line(text) else 0, #check for sig
            1 if is_mostly_internal_whitespace(text) and is_tabular(text) and not is_mostly_sentences(text) and not starts_with_digit_and_special_char(text) else 0, # check for table
            1 if words[0] in common_items_indicators or starts_with_digit(text) else 0, # check for item
            1 if is_all_caps(text) else 0, # check for HEADLINE
            1 if contains_common_address_indicator(text,common_Address_indicators) else 0, # check for address
            1 if is_mostly_sentences(text) else 0, # check for paragraph
            1 if is_mostly_alphanumeric(text) or is_mostly_whitespace(text) or is_mostly_non_alpanum(text) else 0, # checks for graphics
            # 1 if is_bulleted_list(text) else 0,
            # 1 if is_all_caps(text) else 0
            # text.count(' '),
            # sum(1 if w.isupper() else 0 for w in words)
            ]
        else: #format is line
            features = [
            len(text),
            len(text.strip()),
            len(words),
            1 if '>' in words else 0,
            1 if do_most_lines_start_with_quote_char(text, common_QT_indicators) or do_most_lines_start_with_quote_string(text, common_QT_indicators) or words[0] in common_QT_indicators else 0, # check for qt
            1 if words[0].isupper() else 0, # check for nnhead
            1 if words[0] in common_NNHEAD_words else 0, # check for nnhead
            1 if words[0].endswith(':') else 0, # check for nnhead
            1 if is_mostly_internal_whitespace(text) and is_tabular(text) and not is_mostly_sentences(text) and not starts_with_digit_and_special_char(text) else 0, # check for table
            1 if words[0] in common_SIG_indicators or is_single_line(text) else 0, #check for sig
            1 if words[0] in common_items_indicators or starts_with_digit(text) else 0, # check for item
            1 if contains_common_address_indicator(text,common_Address_indicators) else 0, # check for address
            1 if is_all_caps(text) else 0 or is_single_line(text), # check for HEADLINE
            ]

        return features

    
    
    # define a function to check if first word is a date
    # define a function to check for unqiue symbols within the segments
    # define a function to check for number of spaces between words to determine if table tag
    # define a function to check if repeating charcters are present in the segment (if there are it could be a signature)
    # define a function to check if the segment has -- in it (if it does it could be a signature)
    # define a function to check if the first line of the block has a high density of graphical characters (if it does it could be a signature)
    # define a function to check ratio of white space to actual characters in the segment (if it is high it could be a signature or a table or address)
    

    def classify(self, testX):
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)

# load_data function removes #BLANK# and returns X and y
def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            # each line is a tab separated label and text
            arr = line.strip().split('\t', 1) # arr[0] is label, arr[1] is text
            if arr[0] == '#BLANK#': # if label is #BLANK#, skip it
                continue
            X.append(arr[1]) # append text to X
            y.append(arr[0]) # append label to y
        return X, y # X is a list of text, y is a list of labels


def lines2segments(trainX, trainY):
    segX = [] # text
    segY = [] # labels

    #  function that is converting a list of lines and their corresponding labels into a list of segments (groups of consecutive lines with the same label) and their corresponding labels
    for y, group in groupby(zip(trainX, trainY), key=lambda x: x[1]):
        if y == '#BLANK#':  # if the label is #BLANK#, skip it
            continue
        x = '\n'.join(line[0].rstrip('\n') for line in group) # join the lines with the same label
        segX.append(x) # conttains the text of the segment in a single string in the form eg line1\nline2\n, line3\nline4\n
        segY.append(y) # contains the label of the segment in a single string in the form eg label1, label2
    return segX, segY # return the text and labels of the segments


def evaluate(outputs, golds):
    correct = 0
    for h, y in zip(outputs, golds):
        if h == y:
            correct += 1
    print(f'{correct} / {len(golds)}  {correct / len(golds)}')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--format', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs() 
    # args.train, args.test, args.format, args.output, args.errors, args.report
    # extracts the arguments from the command line
    # args.train contains the name of the training file
    # args.test contains the name of the test file
    

    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)
    # trainX is a list of text, trainY is a list of labels
    # testX is a list of text, testY is a list of labels

    if args.format == 'segment': # if the format is segment we need to convert the data to segments by combining adjacent lines with the same label
        trainX, trainY = lines2segments(trainX, trainY) # X is text, Y is label
        testX, testY = lines2segments(testX, testY) # X is text, Y is label
        # testX is a list of segment text, testY is a list of labels

    classifier = SegmentClassifier(args.format)
    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX) 

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    if args.errors is not None:
        with open(args.errors, 'w') as fout:
            for y, h, x in zip(testY, outputs, testX):
                if y != h:
                    print(y, h, x, sep='\t', file=fout)

    if args.report:
        print(classification_report(testY, outputs))
    else:
        evaluate(outputs, testY)


if __name__ == '__main__':
    main()