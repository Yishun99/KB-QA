import os
import codecs
import time
import numpy as np
import tensorflow as tf
import data_util
import similarity
from biLSTM import BiLSTM

# --------------Parameters begin--------------

# data loading params
tf.flags.DEFINE_string("knowledge_file", "data/knowledge.txt", "Knowledge data.")
tf.flags.DEFINE_string("train_file", "data/train.txt", "Training data.")
tf.flags.DEFINE_string("test_file", "data/test.txt", "Test data.")
tf.flags.DEFINE_string("stop_words_file", "data/stop_words.txt", "Stop words.")

# result & model save params
tf.flags.DEFINE_string("result_file", "res/predictRst.score", "Predict result.")
tf.flags.DEFINE_string("save_file", "res/savedModel", "Save model.")

# pre-trained word embedding vectors
tf.flags.DEFINE_string("embedding_file", "/home/douyishun/zhwiki_2017_03.sg_50d.word2vec", "Embedding vectors.")

# hyperparameters
tf.flags.DEFINE_integer("k", 5, "K most similarity knowledge (default: 5).")
tf.flags.DEFINE_integer("rnn_size", 100, "Neurons number of hidden layer in LSTM cell (default: 100).")
tf.flags.DEFINE_float("margin", 0.1, "Constant of max-margin loss (default: 0.1).")
tf.flags.DEFINE_integer("max_grad_norm", 5, "Control gradient expansion (default: 5).")
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50).")
tf.flags.DEFINE_integer("max_sentence_len", 100, "Maximum number of words in a sentence (default: 100).")
tf.flags.DEFINE_float("dropout_keep_prob", 0.45, "Dropout keep probability (default: 0.5).")
tf.flags.DEFINE_float("learning_rate", 0.4, "Learning rate (default: 0.4).")
tf.flags.DEFINE_float("lr_down_rate", 0.5, "Learning rate down rate(default: 0.5).")
tf.flags.DEFINE_integer("lr_down_times", 4, "Learning rate down times (default: 4)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

# training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 20, "Number of checkpoints to store (default: 5)")

# gpu parameters
tf.flags.DEFINE_float("gpu_mem_usage", 0.75, "GPU memory max usage rate (default: 0.75).")
tf.flags.DEFINE_string("gpu_device", "/gpu:0", "GPU device name.")

# misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# --------------Parameters end--------------


# load pre-trained embedding vector
print("loading embedding...")
embedding, word2idx = data_util.load_embedding(FLAGS.embedding_file)

# load stop words
stop_words = codecs.open(FLAGS.stop_words_file, 'r', encoding='utf8').readlines()
stop_words = [w.strip() for w in stop_words]

# top k most related knowledge
print("computing similarity...")
train_sim_ixs = similarity.topk_sim_ix(FLAGS.knowledge_file, FLAGS.train_file, stop_words, FLAGS.k)
test_sim_ixs = similarity.topk_sim_ix(FLAGS.knowledge_file, FLAGS.test_file, stop_words, FLAGS.k)

# --------------Data preprocess begin--------------
print("loading data...")
train_questions, train_answers, train_labels, train_question_num = \
    data_util.load_data(FLAGS.knowledge_file, FLAGS.train_file, word2idx, stop_words, train_sim_ixs, FLAGS.max_sentence_len)

test_questions, test_answers, test_labels, test_question_num = \
    data_util.load_data(FLAGS.knowledge_file, FLAGS.test_file, word2idx, stop_words, test_sim_ixs, FLAGS.max_sentence_len)

#print(train_question_num, len(train_questions), len(train_answers), len(train_labels))
#print(test_question_num, len(test_questions), len(test_answers), len(test_labels))


questions, true_answers, false_answers = [], [], []
for q, ta, fa in data_util.training_batch_iter(
        train_questions, train_answers, train_labels, train_question_num, FLAGS.batch_size):
    questions.append(q), true_answers.append(ta), false_answers.append(fa)
# --------------Data preprocess end--------------


# --------------Training begin--------------
print("training...")
with tf.Graph().as_default(), tf.device(FLAGS.gpu_device):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.gpu_mem_usage
    )
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        gpu_options=gpu_options
    )
    with tf.Session(config=session_conf).as_default() as sess:
        globalStep = tf.Variable(0, name="globle_step", trainable=False)
        lstm = BiLSTM(
            FLAGS.batch_size,
            FLAGS.max_sentence_len,
            embedding,
            FLAGS.embedding_dim,
            FLAGS.rnn_size,
            FLAGS.margin
        )
        
        # define training procedure
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), FLAGS.max_grad_norm)
        saver = tf.train.Saver()

        # output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # summaries
        loss_summary = tf.summary.scalar("loss", lstm.loss)
        summary_op = tf.summary.merge([loss_summary])

        summary_dir = os.path.join(out_dir, "summary", "train")
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        # evaluating
        def evaluate():
            print("evaluating..")
            scores = []
            for test_q, test_a in data_util.testing_batch_iter(test_questions, test_answers, test_question_num, FLAGS.batch_size):
                test_feed_dict = {
                    lstm.inputTestQuestions: test_q,
                    lstm.inputTestAnswers: test_a,
                    lstm.dropout_keep_prob: 1.0
                }
                _, score = sess.run([globalStep, lstm.result], test_feed_dict)
                scores.extend(score)
            cnt = 0
            scores = np.absolute(scores)
            for test_id in range(test_question_num):
                offset = test_id * 4
                predict_true_ix = np.argmax(scores[offset:offset + 4])
                if test_labels[offset + predict_true_ix] == 1:
                    cnt += 1
            print("evaluation acc: ", cnt / test_question_num)

            scores = []
            for train_q, train_a in data_util.testing_batch_iter(train_questions, train_answers, train_question_num, FLAGS.batch_size):
                test_feed_dict = {
                    lstm.inputTestQuestions: train_q,
                    lstm.inputTestAnswers: train_a,
                    lstm.dropout_keep_prob: 1.0
                }
                _, score = sess.run([globalStep, lstm.result], test_feed_dict)
                scores.extend(score)
            cnt = 0
            scores = np.absolute(scores)
            for train_id in range(train_question_num):
                offset = train_id * 4
                predict_true_ix = np.argmax(scores[offset:offset + 4])
                if train_labels[offset + predict_true_ix] == 1:
                    cnt += 1
            print("evaluation acc(train): ", cnt / train_question_num)

        # training
        sess.run(tf.global_variables_initializer())
        lr = FLAGS.learning_rate
        for i in range(FLAGS.lr_down_times):
            optimizer = tf.train.GradientDescentOptimizer(lr)
            optimizer.apply_gradients(zip(grads, tvars))
            trainOp = optimizer.apply_gradients(zip(grads, tvars), global_step=globalStep)
            for epoch in range(FLAGS.num_epochs):
                for question, trueAnswer, falseAnswer in zip(questions, true_answers, false_answers):
                    feed_dict = {
                        lstm.inputQuestions: question,
                        lstm.inputTrueAnswers: trueAnswer,
                        lstm.inputFalseAnswers: falseAnswer,
                        lstm.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    }
                    _, step, _, _, loss, summary = \
                        sess.run([trainOp, globalStep, lstm.trueCosSim, lstm.falseCosSim, lstm.loss, summary_op], feed_dict)
                    print("step:", step, "loss:", loss)
                    summary_writer.add_summary(summary, step)
                    if step % FLAGS.evaluate_every == 0:
                        evaluate()

                saver.save(sess, FLAGS.save_file)
            lr *= FLAGS.lr_down_rate

        # final evaluate
        evaluate()
# --------------Training end--------------
