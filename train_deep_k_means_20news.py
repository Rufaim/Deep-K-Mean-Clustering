import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import cluster_acc
from autoencoder import AutoEncoder
from deep_k_means import DeepKMeans

K = 20
AE_NET = [(n,tf.nn.relu) for n in [500,500,2000]]
EMBEDDING_SIZE = 20
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


_20news = fetch_20newsgroups(data_home="./data")
y_train = _20news.target
vectorizer = TfidfVectorizer(max_features=2000)
X_train = vectorizer.fit_transform(_20news.data)
X_train = X_train.toarray()




ae = AutoEncoder(AE_NET,EMBEDDING_SIZE,SEED)
dkmeans = DeepKMeans(ae,K,seed=SEED)
kmeans = KMeans(n_clusters=K, init="k-means++",random_state=SEED)

logdir = "logs/20news"
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
