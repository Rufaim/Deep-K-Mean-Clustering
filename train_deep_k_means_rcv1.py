import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_rcv1
from utils import print_results, test_averaged_run
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
TEST_RUNS = 10
SEED = 42


gpus = tf.config.list_physical_devices('GPU')
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



def dkmeans_builder(seed):
    ae = AutoEncoder(AE_NET,EMBEDDING_SIZE,seed)
    return DeepKMeans(ae,K,seed=seed)

kmeans_builder = lambda seed: KMeans(n_clusters=K, init="k-means++",random_state=seed)


def dkmeans_fit(alg, data):
    alg.fit(data, BATCH_SIZE, PRETRAIN_EPOCHS, FINETUNE_EPOCH, UPDATE_EPOCH, LEARNING_RATE, LEARNING_RATE, verbose=True)
    cls_dkm, _ = alg(data)
    return cls_dkm.numpy()

def kmeans_fit(alg, data):
    alg.fit(X_train)
    return alg.predict(data)


result_dict = test_averaged_run(dkmeans_builder, kmeans_builder, dkmeans_fit, kmeans_fit, X_train, y_train, TEST_RUNS, SEED)
print_results(result_dict)
