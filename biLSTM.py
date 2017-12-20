import tensorflow as tf


class BiLSTM(object):
    def __init__(self, batch_size, max_sentence_len, embeddings, embedding_size, rnn_size, margin):
        self.batch_size = batch_size
        self.max_sentence_len = max_sentence_len
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.margin = margin

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.inputQuestions = tf.placeholder(tf.int32, shape=[None, self.max_sentence_len])
        self.inputTrueAnswers = tf.placeholder(tf.int32, shape=[None, self.max_sentence_len])
        self.inputFalseAnswers = tf.placeholder(tf.int32, shape=[None, self.max_sentence_len])
        self.inputTestQuestions = tf.placeholder(tf.int32, shape=[None, self.max_sentence_len])
        self.inputTestAnswers = tf.placeholder(tf.int32, shape=[None, self.max_sentence_len])

        # embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            tf_embedding = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            questions = tf.nn.embedding_lookup(tf_embedding, self.inputQuestions)
            true_answers = tf.nn.embedding_lookup(tf_embedding, self.inputTrueAnswers)
            false_answers = tf.nn.embedding_lookup(tf_embedding, self.inputFalseAnswers)

            test_questions = tf.nn.embedding_lookup(tf_embedding, self.inputTestQuestions)
            test_answers = tf.nn.embedding_lookup(tf_embedding, self.inputTestAnswers)
        # LSTM
        with tf.variable_scope("LSTM_scope", reuse=None):
            question1 = self.biLSTMCell(questions, self.rnn_size)
            question2 = tf.nn.tanh(self.max_pooling(question1))
        with tf.variable_scope("LSTM_scope", reuse=True):
            true_answer1 = self.biLSTMCell(true_answers, self.rnn_size)
            true_answer2 = tf.nn.tanh(self.max_pooling(true_answer1))
            false_answer1 = self.biLSTMCell(false_answers, self.rnn_size)
            false_answer2 = tf.nn.tanh(self.max_pooling(false_answer1))

            test_question1 = self.biLSTMCell(test_questions, self.rnn_size)
            test_question2 = tf.nn.tanh(self.max_pooling(test_question1))
            test_answer1 = self.biLSTMCell(test_answers, self.rnn_size)
            test_answer2 = tf.nn.tanh(self.max_pooling(test_answer1))

        self.trueCosSim = self.get_cosine_similarity(question2, true_answer2)
        self.falseCosSim = self.get_cosine_similarity(question2, false_answer2)
        self.loss = self.get_loss(self.trueCosSim, self.falseCosSim, self.margin)

        self.result = self.get_cosine_similarity(test_question2, test_answer2)

    def biLSTMCell(self, x, hidden_size):
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=self.dropout_keep_prob,
                                                     output_keep_prob=self.dropout_keep_prob)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=self.dropout_keep_prob,
                                                     output_keep_prob=self.dropout_keep_prob)
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)
        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])
        return output

    @staticmethod
    def get_cosine_similarity(q, a):
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cosSim = tf.div(mul, tf.multiply(q1, a1))
        return cosSim

    @staticmethod
    def max_pooling(lstm_out):
        height = int(lstm_out.get_shape()[1])
        width = int(lstm_out.get_shape()[2])
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.max_pool(lstm_out, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output

    @staticmethod
    def get_loss(trueCosSim, falseCosSim, margin):
        zero = tf.fill(tf.shape(trueCosSim), 0.0)
        tfMargin = tf.fill(tf.shape(trueCosSim), margin)
        with tf.name_scope("loss"):
            # max-margin losses = max(0, margin - true + false)
            losses = tf.maximum(zero, tf.subtract(tfMargin, tf.subtract(trueCosSim, falseCosSim)))
            loss = tf.reduce_sum(losses)
        return loss
