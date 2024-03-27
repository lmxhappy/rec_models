#encoding=utf-8
import sys
#import weidl_optimizer
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import time



# cgc
LEARNING_RATE = 0.001
BATCH_SIZE=512
SLOT_SIZE = 220
EMB_DIM = 16
INPUT_SIZE = SLOT_SIZE * EMB_DIM
EXPERT_OUTPUT = 200
TOWER_LAYERS_1 = 100
EXPERT_NUM = 1
SELECT_NUM = 2

tf.reset_default_graph()


# Build forward propagation
with tf.name_scope("input"):
    xx = tf.placeholder(tf.float32, shape=[None, SLOT_SIZE, EMB_DIM], name="embeddings")
    x = tf.reshape(xx, [-1, SLOT_SIZE * EMB_DIM])
    y1 = tf.placeholder(tf.float32, shape=[None, 1], name="label1")
    y2 = tf.placeholder(tf.float32, shape=[None, 1], name="label2")
    y3 = tf.placeholder(tf.float32, shape=[None, 1], name="label3")

with tf.name_scope("expert_layers"):
    with tf.name_scope("expert_layer_shared"):
        e1 = tf.Variable(tf.random_normal([INPUT_SIZE, EXPERT_OUTPUT], 0.0, 0.01), name="e1")
        b1 = tf.Variable(tf.random_normal([1, EXPERT_OUTPUT], 0.0, 0.01), name="b1")
        exp_out_shared = tf.add(tf.tensordot(x, e1, axes=1), b1)
        exp_out_shared = tf.nn.relu(exp_out_shared)  #[512,200]

    with tf.name_scope("expert_layer_1"):
        e2 = tf.Variable(tf.random_normal([INPUT_SIZE, EXPERT_OUTPUT], 0.0, 0.01), name="e2")
        b2 = tf.Variable(tf.random_normal([1, EXPERT_OUTPUT], 0.0, 0.01), name="b2")
        exp_out_1 = tf.add(tf.tensordot(x, e2, axes=1), b2)
        exp_out_1 = tf.nn.relu(exp_out_1)

    with tf.name_scope("expert_layer_2"):
        e3 = tf.Variable(tf.random_normal([INPUT_SIZE, EXPERT_OUTPUT], 0.0, 0.01), name="e3")
        b3 = tf.Variable(tf.random_normal([1, EXPERT_OUTPUT], 0.0, 0.01), name="b3")
        exp_out_2 = tf.add(tf.tensordot(x, e3, axes=1), b3)
        exp_out_2 = tf.nn.relu(exp_out_2)

    with tf.name_scope("expert_layer_3"):
        e4 = tf.Variable(tf.random_normal([INPUT_SIZE, EXPERT_OUTPUT], 0.0, 0.01), name="e4")
        b4 = tf.Variable(tf.random_normal([1, EXPERT_OUTPUT], 0.0, 0.01), name="b4")
        exp_out_3 = tf.add(tf.tensordot(x, e4, axes=1), b4)
        exp_out_3 = tf.nn.relu(exp_out_3)

with tf.name_scope("gate_layers"):
# cgc:gate个数等于任务个数
# ple:gate个数等于任务个数+1,即每个任务都有一个gate,share也有一个gate
    # with tf.name_scope("gate_shared"):
    #     g1 = tf.Variable(tf.random_normal([INPUT_SIZE, EXPERT_NUM*(SELECT_NUM+1)], 0.0, 0.01), name="g1")
    #     b5 = tf.Variable(tf.random_normal([1, EXPERT_NUM*(SELECT_NUM+1)], 0.0, 0.01), name="b5")
    #     gate_out_shared = tf.add(tf.matmul(x, g1), b5)
    #     gate_out_shared = tf.nn.softmax(gate_out_shared)  #[512,3]

    with tf.name_scope("gate_1"):
        g2 = tf.Variable(tf.random_normal([INPUT_SIZE, EXPERT_NUM*SELECT_NUM], 0.0, 0.01), name="g2")
        b6 = tf.Variable(tf.random_normal([1, EXPERT_NUM*SELECT_NUM], 0.0, 0.01), name="b6")
        gate_out_1 = tf.add(tf.matmul(x,g2),b6)
        gate_out_1 = tf.nn.softmax(gate_out_1)  #[512,2]

    with tf.name_scope("gate_2"):
        g3 = tf.Variable(tf.random_normal([INPUT_SIZE, EXPERT_NUM*SELECT_NUM], 0.0, 0.01), name="g3")
        b7 = tf.Variable(tf.random_normal([1, EXPERT_NUM*SELECT_NUM], 0.0, 0.01), name="b7")
        gate_out_2 = tf.add(tf.matmul(x,g3),b7)
        gate_out_2 = tf.nn.softmax(gate_out_2)   #[512,2]

    with tf.name_scope("gate_3"):
        g4 = tf.Variable(tf.random_normal([INPUT_SIZE, EXPERT_NUM*SELECT_NUM], 0.0, 0.01), name="g4")
        b8 = tf.Variable(tf.random_normal([1, EXPERT_NUM*SELECT_NUM], 0.0, 0.01), name="b8")
        gate_out_3 = tf.add(tf.matmul(x,g4),b8)
        gate_out_3 = tf.nn.softmax(gate_out_3)   #[512,2]

with tf.name_scope("expert_gate_weight"):
    experts_outputs_1= tf.stack([exp_out_1,exp_out_shared],axis=2)
    expanded_gate_output1 = tf.expand_dims(gate_out_1, axis=1)  # [512,1,2]
    weighted_expert_output1 = experts_outputs_1 * tf.tile(expanded_gate_output1,[1, EXPERT_OUTPUT, 1])  # [512,100,2]
    final_output1 = tf.reduce_sum(weighted_expert_output1, axis=2)  # [512,100]

    experts_outputs_2 = tf.stack([exp_out_2, exp_out_shared], axis=2)
    expanded_gate_output2 = tf.expand_dims(gate_out_2, axis=1)
    weighted_expert_output2 = experts_outputs_2 * tf.tile(expanded_gate_output2, [1, EXPERT_OUTPUT, 1])
    final_output2 = tf.reduce_sum(weighted_expert_output2, axis=2)

    experts_outputs_3 = tf.stack([exp_out_3, exp_out_shared], axis=2)
    expanded_gate_output3 = tf.expand_dims(gate_out_3, axis=1)
    weighted_expert_output3 = experts_outputs_3 * tf.tile(expanded_gate_output3, [1, EXPERT_OUTPUT, 1])
    final_output3 = tf.reduce_sum(weighted_expert_output3, axis=2)

with tf.name_scope("tower_layers_task1"):
    with tf.name_scope("tower_task1_1"):
        t1 = tf.Variable(tf.random_normal([EXPERT_OUTPUT, TOWER_LAYERS_1], 0.0, 0.01), name="t1")
        b9 = tf.Variable(tf.random_normal([1, TOWER_LAYERS_1], 0.0, 0.01), name="b9")
        tower_task1_1 = tf.add(tf.matmul(final_output1,t1),b9)
        tower_task1_1 = tf.nn.relu(tower_task1_1)

    with tf.name_scope("tower_task1_2"):
        t2 = tf.Variable(tf.random_normal([TOWER_LAYERS_1, 1], 0.0, 0.01), name="t2")
        b10 = tf.Variable(tf.random_normal([1, 1], 0.0, 0.01), name="b10")
        task1_out = tf.add(tf.matmul(tower_task1_1,t2),b10)


with tf.name_scope("tower_layers_task2"):
    with tf.name_scope("tower_task2_1"):
        t3 = tf.Variable(tf.random_normal([EXPERT_OUTPUT, TOWER_LAYERS_1], 0.0, 0.01), name="t3")
        b11= tf.Variable(tf.random_normal([1, TOWER_LAYERS_1], 0.0, 0.01), name="b11")
        tower_task2_1 = tf.add(tf.matmul(final_output2,t3),b11)
        tower_task2_1 = tf.nn.relu(tower_task2_1)

    with tf.name_scope("tower_task2_2"):
        t4 = tf.Variable(tf.random_normal([TOWER_LAYERS_1, 1], 0.0, 0.01), name="t4")
        b12 = tf.Variable(tf.random_normal([1, 1], 0.0, 0.01), name="b12")
        task2_out = tf.add(tf.matmul(tower_task2_1,t4),b12)

with tf.name_scope("tower_layers_task3"):
    with tf.name_scope("tower_task3_1"):
        t5 = tf.Variable(tf.random_normal([EXPERT_OUTPUT, TOWER_LAYERS_1], 0.0, 0.01), name="t5")
        b13= tf.Variable(tf.random_normal([1, TOWER_LAYERS_1], 0.0, 0.01), name="b13")
        tower_task3_1 = tf.add(tf.matmul(final_output3,t5),b13)
        tower_task3_1 = tf.nn.relu(tower_task3_1)

    with tf.name_scope("tower_task3_2"):
        t6 = tf.Variable(tf.random_normal([TOWER_LAYERS_1, 1], 0.0, 0.01), name="t6")
        b14 = tf.Variable(tf.random_normal([1, 1], 0.0, 0.01), name="b14")
        task3_out = tf.add(tf.matmul(tower_task3_1,t6),b14)

loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y1, logits=task1_out)
loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y2, logits=task2_out)
loss3 = tf.square(y3-task3_out)
loss = tf.concat([loss1,loss2,0.5 * loss3],axis=1)
predict_loss_vec = tf.reduce_mean(loss,axis=0)
# predict_loss_vec = tf.div(tf.reduce_sum(tf.multiply(loss,tags),axis=0),tf.reduce_sum(tags,axis=0))
loss_var1 = tf.Variable([0.0], name="loss_var1")
loss_var2 = tf.Variable([0.0], name="loss_var2")
loss_var3 = tf.Variable([0.0], name="loss_var3")
predict_loss = 2 * predict_loss_vec[0] * tf.exp(-loss_var1) + 2 * predict_loss_vec[1] * tf.exp(-loss_var2) + predict_loss_vec[2] * tf.exp(-loss_var3) + loss_var1 + loss_var2 + loss_var3

pre_loss1 = tf.identity(predict_loss_vec[0],name="pre_loss1")
pre_loss2 = tf.identity(predict_loss_vec[1],name="pre_loss2")
pre_loss3 = tf.identity(predict_loss_vec[2],name="pre_loss3")
predict_loss = tf.identity(predict_loss,name="predict_loss")

predict_out1 = tf.sigmoid(task1_out,name="predict_out1")
predict_out2 = tf.sigmoid(task2_out,name="predict_out2")
predict_out3 = tf.identity(task3_out,name="predict_out3")

label1 = tf.identity(y1,"label1")
label2 = tf.identity(y2,"label2")
label3 = tf.identity(y3,"label3")

auc_1 = tf.metrics.auc(labels=y1, predictions=predict_out1, num_thresholds=2000)
auc_2 = tf.metrics.auc(labels=y2, predictions=predict_out2, num_thresholds=2000)
auc_out1 = tf.identity(auc_1, name="auc_out1")
auc_out2 = tf.identity(auc_2, name="auc_out2")


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-6,name="Adam")
    train_op = optimizer.minimize(predict_loss)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name="weidl_init")


# Start run
with tf.Session() as sess:
    sess.run(init)
    tf.train.write_graph(sess.graph_def, "./", "online_cgc_model.pb", as_text=False)