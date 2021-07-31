import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.datasets import fetch_rcv1
from utils import cluster_acc
from autoencoder import AutoEncoder
from deep_k_means import DeepKMeans

K = 4
AE_NET = [(n,tf.nn.relu) for n in [500,500,2000]]
EMBEDDING_SIZE = 4
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
PRETRAIN_EPOCHS = 500
FINETUNE_EPOCH = 250
UPDATE_EPOCH = 1
SEED = 42


gpus = tf.config.list_physical_devices('GPU')
# Out of jokes it is highly recommended to run the following experiment on gpu
for gpu in gpus:
    tf.config.set_logical_device_configuration(gpu,
    [tf.config.LogicalDeviceConfiguration(memory_limit=2*1024)])


rcv1 = fetch_rcv1(subset="all", data_home="./data", shuffle=True)
data = rcv1.data[:10000] # sample 10000 documents
target = rcv1.target[:10000]
sum_tfidf = np.asarray(sp.spmatrix.sum(data, axis=0))[0] # Sum of tf-idf for all words based on the filtered dataset
word_indices = np.argpartition(-sum_tfidf, 2000)[:2000] # Keep only the 2000 top words in the vocabulary
data = data[:, word_indices].toarray()


names = rcv1.target_names
category_names = ['CCAT', 'ECAT', 'GCAT', 'MCAT']
top_four_classes_indeces = [i for i in range(len(names)) if names[i] in category_names]
top_four_classes_indeces = np.array(top_four_classes_indeces,dtype=np.int)
top_four_classes_data_indeces = np.array(np.sum(target[:,top_four_classes_indeces], axis=-1) == 1,dtype=np.bool)[:,0]
y_train = target[top_four_classes_data_indeces]
y_train = y_train[:,top_four_classes_indeces].toarray()
y_train = np.sum(y_train * np.arange(4),axis=-1)
X_train = data[top_four_classes_data_indeces]



ae = AutoEncoder(AE_NET,EMBEDDING_SIZE,SEED)
dkmeans = DeepKMeans(ae,K,seed=SEED)
kmeans = KMeans(n_clusters=K, init="k-means++",random_state=SEED)

logdir = "logs/rcv1"
file_writer = tf.summary.create_file_writer(logdir,flush_millis=10000)
file_writer.set_as_default()

dkmeans.fit(X_train,BATCH_SIZE,PRETRAIN_EPOCHS,FINETUNE_EPOCH,UPDATE_EPOCH,LEARNING_RATE,LEARNING_RATE,seed=SEED,verbose=True)
kmeans.fit(X_train)

cls_dkm,_ = dkmeans(X_train)
cls_km = kmeans.predict(X_train)

print("K-means")
print("   ACC: ", cluster_acc(y_train,cls_km))
print("   NMI: ", normalized_mutual_info_score(y_train,cls_km))
print("Deep K-means")
print("   ACC: ", cluster_acc(y_train,cls_dkm.numpy()))
print("   NMI: ", normalized_mutual_info_score(y_train,cls_dkm.numpy()))
