import os
import re
import tqdm
import logging
import random
import numpy as np
import unicodedata
import argparse
import torch
import torch.nn as nn

from torch.autograd import Variable
from RNN_model import RNN

EOS = "#EOS"


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def load_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            line = line.replace("\n", "").strip()
            if len(line) > 0:
                data.append(line)

    logging.info("Length of Data: {} \n".format(len(data)))
    logging.info("Random Text: {}".format(data[random.randint(0, len(data))]))
    return data


def get_all_characters(list_text: list):
    text_joined = ''.join(list_text)
    list_characters = list(set(text_joined))
    logging.info("{} characters".format(len(list_characters)))
    return list_characters


def get_char_index(c):
    if c not in all_characters:
        raise ValueError("{} is not a character available !".format(c))
    return all_characters.index(c)


def text_to_one_hot_vector(text):
    tensor = torch.zeros(len(text), 1, n_characters)
    for i, c in enumerate(text):
        try:
            tensor[i][0][get_char_index(c)] = 1
        except:
            tensor[i][0][get_char_index(c)] = 0
    return Variable(tensor)


def text_to_expected_input(text):
    y = []
    # We start to 1 because the first character is not predicted
    for c in text[1:]:
        y.append(get_char_index(c))
    # We add the End of String Element
    y += [n_characters - 1]
    y = Variable(torch.LongTensor(y))
    x = text_to_one_hot_vector(text)
    return x, y


def generate(text_start, predict_len=300):
    hidden = rnn.init_hidden()
    start_input, start_expected = text_to_expected_input(text_start)

    # We start by learning the hidden layer from the start text
    for inp in start_input[:-1]:
        _, hidden = rnn(inp, hidden)
    new_inp = start_input[-1]

    i = 0
    predicted_char = ""
    predicted = text_start
    while predicted_char != EOS and i < predict_len:
        output, hidden = rnn(new_inp, hidden)

        top_i = torch.topk(output, 1)[1].data.tolist()[0][0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        new_inp = text_to_one_hot_vector(predicted_char)[0]
        i += 1

    return predicted


def train(n_epochs=1000):
    all_losses = []
    loss_avg = 0

    list_random = list(np.random.randint(0, len(data_vectors), n_epochs))
    for epoch in tqdm.tqdm(range(0, n_epochs)):
        rand = list_random[epoch]

        x = data_vectors[rand]["x"]
        y = data_vectors[rand]["y"]

        hidden = rnn.init_hidden()
        rnn.zero_grad()

        loss = 0
        for i, elem in enumerate(x):
            output, hidden = rnn(elem, hidden)
            loss += criterion(output, y[i])

        loss.backward()
        optimizer.step()

        loss_avg += loss.data[0] / x.size()[-1]

        if epoch % 1000 == 0:
            print('Epochs: {}'.format(epoch))
            print(generate('hi', 1000), '\n')

        if epoch % 100 == 0:
            all_losses.append(loss_avg / 100)
            loss_avg = 0
    return all_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', '-fp', help='Input path file. A txt file is expected')
    parser.add_argument('--verbose', '-v', default=10, help='Verbose Level')

    parser.add_argument('--n_epochs', '-ep', default=1000, help='Number of epochs to do')
    parser.add_argument('--num_layers', default=1, help='Number of Layer for RNN')
    parser.add_argument('--bidirectional', default=False, help='Make bidirectional RNN')
    parser.add_argument('--hidden_size', default=100, help='RNN hidden size')
    parser.add_argument('--model', default="LSTM", help='RNN Model to use. Model Available: LSTM, RNN, GRU')

    args = parser.parse_args()

    logging.getLogger().setLevel(args.verbose)
    logging.info("Loading Data")
    data_text = load_data(args.filepath)

    logging.info("Getting All Characters")
    all_characters = get_all_characters(data_text)
    n_characters = len(all_characters)

    logging.info("Encoding Vectors")
    data_vectors = []
    for i, text in enumerate(data_text):
        x, y = text_to_expected_input(text)
        data_vectors.append({
            'index_text': i,
            'x': x,
            'y': y
        })

    logging.info("Define RNN instance")
    rnn = RNN(input_size=n_characters,
              hidden_size=args.hidden_size,
              output_size=n_characters,
              num_layers=args.num_layers,
              bidirectional=args.bidirectional,
              model_type=args.model,
              dropout=0.5)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    logging.info("Start Training")
    train(n_epochs=args.n_epochs)