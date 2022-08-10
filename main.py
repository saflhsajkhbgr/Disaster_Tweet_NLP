# -*- coding:utf-8 -*-
"""
@author: Felix Z
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import gensim
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
import re
import matplotlib.pyplot as plt


class NBclassifier:
    def __init__(self, clf_path=None, vec_path=None):
        if clf_path == None or vec_path == None:
            self.clf = MultinomialNB()
            self.vec = TfidfVectorizer()
        else:
            self.clf = joblib.load(clf_path)
            self.vec = joblib.load(vec_path)

    def saveModel(self, clf_path="clf.m", vec_path="vec.m"):
        joblib.dump(self.clf,clf_path)
        joblib.dump(self.vec,vec_path)

    def trainNB(self, dataList,labelList):
        self.clf.fit(self.vec.fit_transform(dataList), labelList)
        self.saveModel(clf_path='clf.m',
                       vec_path='vec.m')

    def predictNB(self, dataList):
        data = self.vec.transform(dataList)
        predictList=self.clf.predict(data)
        return predictList

    def calAccuracy(self, labelList, predictList):
        rightCount=0
        if(len(labelList)==len(predictList)):
            for i in range(len(labelList)):
                if(labelList[i]==predictList[i]):
                    rightCount+=1
            accuracy = rightCount/float(len(labelList))
            print('准确率为：%s' %(rightCount/float(len(labelList))))
            return accuracy


# def main():
#     classifier = NBclassifier()
#     trainsingset, testset, train_kv = classifier.load_data()
#     iterations = 10
#
#     for i in range(iterations):
#         train_kv = shuffle(train_kv)
#         classifier.trainNB(trainsingset[0], trainsingset[1])
#         pred = classifier.predictNB(testset[0])
#         classifier.saveModel()
#         if classifier.calAccuracy(testset[1], pred) > 0.9:
#             break


# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)
# print(x_train.shape, len(x_train[0]), y_train.shape)
# print(x_test.shape, len(x_test[0]), y_test.shape)
# print(x_train[0])


def load_data():
    """
    :return: training set, test set
    """
    tweets = []
    labels = []
    df = pd.DataFrame(pd.read_csv('Disaster Tweet/train.csv'))
    # print(df.items)
    for i in range(len(df)):
        tweets.append(df['text'][i])
        labels.append(df['target'][i])

    return tweets, labels


# sequence = '#news Twelve feared killed in Pakistani air ambulance helicopter crash http://t.co/bFeS5tWBzt #til_now #DNA'
#
# lst_of_words = sequence.split()
# for j in range(len(lst_of_words)):
#     if '#' in lst_of_words[j]:
#         lst_of_words[j] = re.compile('#(.*)').findall(lst_of_words[j])[0]
#     if '@' in lst_of_words[j]:
#         lst_of_words[j] = re.compile('@(.*)').findall(lst_of_words[j])[0]
#     if 'http' in lst_of_words[j]:
#         lst_of_words[j] = ''
# print(lst_of_words)


class Word2Vec:
    """
    Embedding 层本身是一个字典， 根据下标去寻找对应的词向量映射
    对于自己的训练集则需要构造自己的word_index
    """
    def __init__(self, weight_path, embed_dim, trainable=False):
        """
        self.weight_path: word2vec weights
        self.embed_dim: default 300
        self.trainable: Depends, default FALSE
        self.vocab: {word: index}
        self.word_index: [(word, vector)...]
        self.embedding_matrix: weight for the keras.Embedding
        """
        self.trainable = trainable
        self.weight_path = weight_path
        self.embed_dim = embed_dim
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.weight_path, binary=True)

    def __call__(self, *args, **kwargs):
        self.vocab = {'PAD': 0}
        self.word_index = [(k, self.model.wv[k]) for k, v in self.model.wv.vocab.items()]
        self.embeddings_matrix = np.zeros((len(self.model.wv.vocab.items()) + 1, self.model.vector_size))
        for i in range(len(self.word_index)):
            self.vocab[self.word_index[i][0]] = i + 1
            self.embeddings_matrix[i + 1] = self.word_index[i][1]
        print('vocab dict:', list(self.vocab.keys())[0:5], list(self.vocab.values())[0:5])
        print('word vec kv:', self.word_index[0:5])
        print('matrix:', self.embeddings_matrix[0:5])

    def get_layer(self):
        embedding = tf.keras.layers.Embedding(input_dim=len(self.embeddings_matrix),
                                              output_dim=self.embed_dim,
                                              weights=[self.embeddings_matrix],
                                              trainable=False)
        return embedding

    def get_most_similar(self, word):
        items = self.model.most_similar(word)
        for item in items:
            print(item[0], item[1])
        return items

    def get_similarity(self, word1, word2):
        return self.model.similarity(word1, word2)


# model = Word2Vec(weight_path='GoogleNews-vectors-negative300.bin', embed_dim=300)


def preprocessing(x, y, vocabdict):
    """
    vocabdict: word2vec vocab dictionary with {word: index}
    1. calculate mean length to determine maximum sentence length
    2. get words out of sentence by regular expressions
    3. 打平矩阵
    :return: shape = [b, sequence length]
    """
    total = 0
    counter = 0
    for sequence in range(len(x)):
        lst_of_words = x[sequence].split()
        print('lst of words:', lst_of_words)
        # dealing with hashtags, @, ., ?, !, and websites
        for j in range(len(lst_of_words)):
            if '?' in lst_of_words[j]:
                lst_of_words[j] = re.compile('(.*)?').findall(lst_of_words[j])[0]
            if '!' in lst_of_words[j]:
                lst_of_words[j] = re.compile('(.*)!').findall(lst_of_words[j])[0]
            if '.' in lst_of_words[j]:
                lst_of_words[j] = re.compile('(.*).').findall(lst_of_words[j])[0]
            if '#' in lst_of_words[j]:
                lst_of_words[j] = re.compile('#(.*)').findall(lst_of_words[j])[0]
            if '@' in lst_of_words[j]:
                lst_of_words[j] = re.compile('@(.*)').findall(lst_of_words[j])[0]
            if 'http' in lst_of_words[j]:
                lst_of_words[j] = ''
        print(lst_of_words)

        # finding the index for each words
        index_list = []
        for i in range(len(lst_of_words)):
            try:
                index_list.append(vocabdict[lst_of_words[i]])
            except KeyError:
                print('not in vocab:', lst_of_words[i])
                pass
        print(index_list)

        # iterations
        total += len(index_list)
        counter += 1
        print(counter, 'out of', len(x))
        x[sequence] = index_list
        print('-'*100)

    avglength = total/len(x)
    return x, y, avglength


class GRUClassifier(tf.keras.Model):
    def __init__(self, units, word2vec):
        super(GRUClassifier, self).__init__()

        # [b, 28]，构建Cell初始化状态向量，重复使用
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]
        self.state2 = [tf.zeros([batchsz, units])]
        self.state3 = [tf.zeros([batchsz, units])]

        # 词向量编码 [b, 80] => [b, 12, 300]
        self.embedding = word2vec.get_layer()

        # 构建Cell
        self.rnn_cell0 = tf.keras.layers.GRUCell(units, dropout=0.3)
        self.rnn_cell1 = tf.keras.layers.GRUCell(units, dropout=0.3)
        self.rnn_cell2 = tf.keras.layers.GRUCell(units, dropout=0.3)
        self.rnn_cell3 = tf.keras.layers.GRUCell(units, dropout=0.3)
        # 构建分类网络，用于将CELL的输出特征进行分类，2分类
        # [b, 12, 300] => [b, 28] => [b, 1]
        self.outlayer = tf.keras.Sequential([
            tf.keras.layers.Dense(units),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1)])

    def call(self, inputs, training=None):
        x = inputs # [b, 12]
        # embedding: [b, 12] => [b, 12, 300]
        x = self.embedding(x)
        # rnn cell compute,[b, 12, 300] => [b, 28]
        state0 = self.state0
        state1 = self.state1
        state2 = self.state2
        state3 = self.state3
        for word in tf.unstack(x, axis=1): # word: [b, 28]
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
            out2, state2 = self.rnn_cell2(out1, state2, training)
            out3, state3 = self.rnn_cell3(out2, state3, training)
        # 末层最后一个输出作为分类网络的输入: [b, 28] => [b, 1]
        x = self.outlayer(out3, training)
        # p(y is pos|x)
        prob = tf.sigmoid(x)
        return prob


class TextCNN(tf.keras.Model):
    def __init__(self, word2vec):
        super(TextCNN, self).__init__()

        self.word2vec = word2vec.get_layer()
        self.conv1 = keras.layers.Conv1D(filters=5, kernel_size=2, padding='valid', strides=1, activation='relu')
        self.conv2 = keras.layers.Conv1D(filters=5, kernel_size=3, padding='valid', strides=1, activation='relu')
        self.conv3 = keras.layers.Conv1D(filters=5, kernel_size=4, padding='valid', strides=1, activation='relu')
        self.conv4 = keras.layers.Conv1D(filters=5, kernel_size=5, padding='valid', strides=1, activation='relu')
        self.conv5 = keras.layers.Conv1D(filters=5, kernel_size=6, padding='valid', strides=1, activation='relu')

        self.concate = keras.layers.Concatenate(axis=-1)
        self.flatten = keras.layers.Flatten()

        self.dense = keras.layers.Dense(64, activation=tf.nn.relu)
        self.dropout = keras.layers.Dropout(0.25)
        self.logits = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        x = self.word2vec(inputs)
        x1 = self.conv1(x)
        x1 = keras.layers.MaxPool1D(pool_size=int(x1.shape[1]))(x1)
        x2 = self.conv2(x)
        x2 = keras.layers.MaxPool1D(pool_size=int(x2.shape[1]))(x2)
        x3 = self.conv3(x)
        x3 = keras.layers.MaxPool1D(pool_size=int(x3.shape[1]))(x3)
        x4 = self.conv4(x)
        x4 = keras.layers.MaxPool1D(pool_size=int(x4.shape[1]))(x4)
        x5 = self.conv5(x)
        x5 = keras.layers.MaxPool1D(pool_size=int(x5.shape[1]))(x5)

        x = self.flatten(self.concate([x1, x2, x3, x4, x5]))
        x = self.dropout(self.dense(x))
        x = self.logits(x)
        return x


def train():
    # 加载Word2Vec词向量模型
    word2vec = Word2Vec(weight_path='GoogleNews-vectors-negative300.bin', embed_dim=embedding_len)
    x, y = load_data()
    word2vec()

    # 数据预处理
    x, y, avglength = preprocessing(x, y, word2vec.vocab)
    print(avglength)

    # 准备数据
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=max_review_len)
    x_train, x_test = x[0:split], x[split:]
    y_train, y_test = y[0:split], y[split:]
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.batch(batchsz, drop_remainder=True)
    print(x_train[0:5])
    print(x_train.shape)
    print(x_test[0:5])
    print(x_test.shape)

    # tensorboard 可视化
    tbcallback = tf.keras.callbacks.TensorBoard(update_freq='batch', write_graph=True, write_images=True)

    # GRU Model
    epochs = 40
    model = GRUClassifier(units=8, word2vec=word2vec)
    model.compile(optimizer=tf.optimizers.RMSprop(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(db_train, epochs=epochs, validation_data=db_test)
    model.evaluate(db_test)

    # Text CNN model
    # epochs = 5
    # model = TextCNN(word2vec)
    # model.compile(optimizer=tf.optimizers.RMSprop(0.001),
    #               loss=tf.losses.BinaryCrossentropy(),
    #               metrics=['accuracy'])
    # history = model.fit(db_train, epochs=epochs, validation_data=db_test, callbacks=[tbcallback])
    # model.evaluate(db_test)

    # 可视化
    ig1, ax_acc = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'])
    print('accuracy:', history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    print('val_accuracy:', history.history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'],
               loc='upper right')
    plt.show()

    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.show()

    dec = input('Submit?')
    if dec == 'y':
        # 数据预处理
        ids = []
        tweets = []
        df = pd.DataFrame(pd.read_csv('Disaster Tweet/test.csv'))
        for i in range(len(df)):
            tweets.append(df['text'][i])
            ids.append(df['id'][i])
        x, y, avglength = preprocessing(tweets, None, word2vec.vocab)

        # 准备数据
        x = keras.preprocessing.sequence.pad_sequences(x, maxlen=max_review_len)
        print(x)

        # GRU
        predict = []
        for i in range(len(x)):
            result = model.predict(tf.reshape(x[i], (1, 30)))
            predict.append(result)

        dic = {'id': [], 'target': []}
        for i in range(len(ids)):
            dic['id'].append(ids[i])
            if predict[i][0][0] > 0.5:
                dic['target'].append(1)
            else:
                dic['target'].append(0)
        df = pd.DataFrame(dic)
        df.to_csv('submission1.csv', index=False)
    dec = input('save weight?')
    if dec == 'y':
        model.save_weights('Disaster_Tweet_GRU_Weight.h5', save_format='h5')


split = 6000
batchsz = 128 # 批量大小
max_review_len = 30 # 句子最大长度s，大于的句子部分将截断，小于的将填充
embedding_len = 300 # 词向量特征长度f

# (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
# print(x_train[0:5])
# print(x_test[0:5])


def prediction():
    # 加载Word2Vec词向量模型
    word2vec = Word2Vec(weight_path='GoogleNews-vectors-negative300.bin', embed_dim=embedding_len)
    word2vec()

    # 数据预处理
    ids = []
    tweets = []
    df = pd.DataFrame(pd.read_csv('Disaster Tweet/test.csv'))
    for i in range(len(df)):
        tweets.append(df['text'][i])
        ids.append(df['id'][i])
    x, y, avglength = preprocessing(tweets, None, word2vec.vocab)

    # 准备数据
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=max_review_len)
    print(x)

    # Text CNN
    # model = TextCNN(word2vec)
    # model(x)
    # model.load_weights('Disaster_Tweet_TextCNN_Weight.h5')
    # predict = model.predict(x)

    # GRU
    model = GRUClassifier(units=8, word2vec=word2vec)
    model(x[0:128])
    model.load_weights('Disaster_Tweet_GRU_Weight.h5')
    predict = model.predict(x)

    dic = {'id': [], 'target': []}
    for i in range(len(ids)):
        dic['id'].append(ids[i])
        if predict[i].any() > 0.5:
            dic['target'].append(1)
        else:
            dic['target'].append(0)
    df = pd.DataFrame(dic)
    df.to_csv('submission.csv', index=False)


def compare():
    lst1 = pd.read_csv('submission1.csv')
    df1 = pd.DataFrame(lst1)
    lst2 = pd.read_csv('submission2.csv')
    df2 = pd.DataFrame(lst2)
    for i in range(len(df1)):
        if df1['target'][i] != df2['target'][i]:
            print(df1['id'][i])


if __name__ == '__main__':
    train()
    # prediction()
    # compare()