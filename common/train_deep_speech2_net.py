
# coding: utf-8

# In[1]:

import data_utils
from utils import Dataset,ProgressBar
import tflearn
from tflearn.data_flow import DataFlow,DataFlowStatus,FeedDictFlow
#from model import DeepSpeech,supported_rnns
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import re


# In[3]:

np.zeros([3,2]).size


# ## The implemention of pytorch's DataLoader is not competiable with tflearn's DataFlow
# ```python
# train_loader = data_utils.AudioDataLoader(
#     data_utils.SpectrogramDataset(manifest_filepath='./data/ted_train_manifest_fix.csv')
#     ,batch_size=16,shuffle=True)
# 
# test_loader = data_utils.AudioDataLoader(
#     data_utils.SpectrogramDataset(manifest_filepath='./data/ted_test_manifest_fix.csv')
#     ,batch_size=16,shuffle=True)
# ```

# # CONFIG

# In[5]:


SAMPLE_RATE = 16000
WINDOW_SIZE = .02
WINDOW_STRIDE = .01
WINDOW = 'hamming'

HIDDEN_LAYERS = 2
#RNN_TYPE = supported_rnns['gru']
BIDIRECTIONAL = True

BEGINING_LEARNING_RATE = 1e-5
MOMENTUM = 0.9
MAX_NORM = 400
#LABELS = ''.join(data_utils.LABELS)
BATCH_SIZE = 2


GPU_CORE = 0
RNN_SIZE = 300#768
MODEL_NAME = "11_18_deepspeech2"
#audio_conf = dict(sample_rate=SAMPLE_RATE,
#                  window_size=WINDOW_SIZE,
#                  window_stride=WINDOW_STRIDE,
#                  window=WINDOW,
#                )


# In[6]:

voice_dataset_train = data_utils.VoicesFlow(manifest_filepath='./data/ted_train_manifest_fix.csv')
voice_flow_train = data_utils.get_flow(voice_dataset_train,shuffle=True,batch_size=BATCH_SIZE)

voice_dataset_test = data_utils.VoicesFlow(manifest_filepath='./data/ted_test_manifest_fix.csv')
voice_flow_test = data_utils.get_flow(voice_dataset_test,shuffle=False,batch_size=BATCH_SIZE)


# In[7]:

for i in range(10):
    x,y,a,t,s = voice_flow_train.next(1)['data']


# In[8]:

len(x)


# In[9]:

index = 0
x.shape,len(y),len(y[0])


# In[14]:

#转化一个序列列表为稀疏矩阵    
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    
    
    return indices, values, shape


# In[15]:

x.shape


# In[16]:

def conv_layer(input_tensor,training,kernel_size=(3,3),filters=32,dropout=None
               ,clip_net=20,batch_normalization=True,strides=(1,1,1,1)):
    net = tf.layers.conv2d(input_tensor,filters=filters,kernel_size=kernel_size,padding='same',strides=strides)
    if batch_normalization:
        net = tf.layers.batch_normalization(net,training=training)
    if dropout is not None:
        net = tf.layers.dropout(net,dropout=1.-dropout)
    net = tf.nn.relu(net)
    #if clip_net is not None and clip_net != False and clip_net > 0:
    #    net = tf.minimum(net,tf.Variable(clip_net))
    return net

def rnn_layer(input_tensor,layer_num,rnn_seq_lens):
    forward_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(RNN_SIZE) for i in range(layer_num)])
    backward_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(RNN_SIZE) for i in range(layer_num)])
    #bi_outputs, bi_encoder_state,_ = tf.contrib.rnn.static_bidirectional_rnn(
    #        forward_cell,backward_cell,tf.unstack(net,net_shape[1],1),dtype=tf.float32)
    bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                forward_cell, backward_cell, input_tensor,
                sequence_length = rnn_seq_lens,time_major = False,dtype=tf.float32)
    outputs_fw, outputs_bw = bi_outputs
    rnn_outputs = outputs_fw + outputs_bw
    return rnn_outputs
    


# In[17]:

def _activation_summary(act,tensor_name):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      act: Tensor
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tf.summary.histogram(tensor_name + '/activations', act)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(act))


# In[18]:

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

with tf.device("/gpu:{}".format(GPU_CORE)):
    input_sound = tf.placeholder(tf.float32,shape=[BATCH_SIZE,161,None,1])
    # [batch_size,freq,time,1]
    output_text = tf.sparse_placeholder(tf.int32)
    voicelength = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
    seqlength = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
    global_step = tf.train.get_or_create_global_step()
    max_grad = tf.placeholder(tf.float32)
    
    targets = tf.sparse_placeholder(tf.int32)#, shape=np.array([BATCH_SIZE,4], dtype=np.int64))
    learning_rate = tf.placeholder(tf.float32)
    training=tf.placeholder(tf.bool)
    
    conv1 = conv_layer(input_sound,training,kernel_size=(41,11),filters=32,strides=(2,2))
    conv2 = conv_layer(conv1,training,kernel_size=(21,11),filters=32,strides=(2,1))
    conv3 = conv_layer(conv2,training,kernel_size=(21,11),filters=32,strides=(2,1))# filters=96 according to paper
    
    
    net = conv3
    net = tf.transpose(net,[0,2,1,3])
    net_shape = net.get_shape().as_list()
    net = tf.reshape(net,[BATCH_SIZE,-1,net_shape[-1]])
    # [batch_size,time,freq * filters]
    rnn_out = rnn_layer(net,HIDDEN_LAYERS,voicelength)
    _activation_summary(rnn_out,'rnn_output')
    net_result = tf.layers.dense(rnn_out,len(data_utils.LABELS) + 2,activation=None)
    _activation_summary(net_result,'net_result')
    
    with tf.variable_scope("Loss"):
        ctc_loss = tf.nn.ctc_loss(labels=targets,inputs=net_result,sequence_length=seqlength
                                  ,time_major=False)
        loss = tf.reduce_mean(ctc_loss)
        tf.summary.scalar("Loss",loss)

    with tf.variable_scope("LearningRate"):
        tf.summary.scalar("LearningRate",learning_rate)

        
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(tf.transpose(net_result, (1, 0, 2)),
                                                      sequence_length=seqlength,merge_repeated=True)
    dense_decoded = tf.cast(tf.sparse_tensor_to_dense(decoded[0], default_value=0), tf.int32)
    
    with tf.variable_scope("EditDistance"):
        dis = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
        tf.summary.scalar("EditDistance",dis)
        
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(ctc_loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, max_grad)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #train_op = optimizer.minimize(loss,global_step=global_step)
    train_op = optimizer.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=global_step)
    
    summary_op = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter("./log/compair/{}".format(MODEL_NAME), sess.graph)
   
    with tf.variable_scope("Validation"):
        test_loss = tf.placeholder(tf.float32)
        loss_op = tf.summary.scalar("Accuracy",test_loss)
        editdistance = tf.placeholder(tf.float32)
        editdistance_op = tf.summary.scalar("EditDistance",editdistance)
    test_op = tf.summary.merge([loss_op,editdistance_op])
    


# In[19]:

sess.run(tf.global_variables_initializer())
tf.train.global_step(sess, global_step)


# In[20]:

import os
if not os.path.exists("models/{}".format(MODEL_NAME)):
    os.mkdir("models/{}".format(MODEL_NAME))


# In[21]:

saver = tf.train.Saver()
saver.save(sess,'./models/{}/sample'.format(MODEL_NAME))


# In[26]:


# In[27]:

N_BATCH = 10000
N_BATCH_TEST = 200


# In[ ]:

restore = False
N_EPOCH = 50
DECAY_EPOCH = 10

class ExpVal:
    def __init__(self,exp_a=0.97):
        self.val = None
        self.exp_a = exp_a
    def update(self,newval):
        if self.val == None:
            self.val = newval
        else:
            self.val = self.exp_a * self.val + (1 - self.exp_a) * newval
    def getval(self):
        return round(self.val,2)
    
expdis = ExpVal()
exploss = ExpVal()


begining_learning_rate = BEGINING_LEARNING_RATE
clip_norm = 400

if restore == False:
    train_epoch = 1
    train_batch = 0
for one_epoch in range(train_epoch,N_EPOCH):
    train_epoch = one_epoch
    pb = ProgressBar(worksum=N_BATCH * BATCH_SIZE,info=" epoch {} batch {}".format(train_epoch,train_batch))
    pb.startjob()
    
    for one_batch in range(N_BATCH):
        if restore == True and one_batch < train_batch:
            pb.auto_display = False
            pb.complete(BATCH_SIZE)
            pb.auto_display = True
            continue
        else:
            restore = False
        train_batch = one_batch
        
        batch_x,batch_y_ori,a,t,s = voice_flow_train.next()['data']
        batch_y = sparse_tuple_from(batch_y_ori)
        batch_seq_len  = [i // 2 for i in s]
        batch_target_len = [len(i) for i in batch_y_ori]
        
        
        # learning rate decay strategy
        batch_lr = begining_learning_rate * 10 ** -(one_epoch // DECAY_EPOCH)
        
        step_dis,_,step_loss,step_summary,step_value = sess.run(
            [dis,train_op,loss,summary_op,global_step],feed_dict={
                input_sound:batch_x,targets:batch_y,learning_rate:batch_lr,voicelength:batch_seq_len,seqlength:batch_target_len,
                max_grad:clip_norm,training:True
            })
        
        expdis.update(step_dis)
        exploss.update(step_loss)
        pb.info = "EPOCH {} STEP {} LOSS {} DIS {} ".format(one_epoch,one_batch,exploss.getval(),expdis.getval())
        train_summary_writer.add_summary(step_summary,step_value)
        pb.complete(BATCH_SIZE)

    print()
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess,"models/{}/model_{}".format(model_name,one_epoch))



