import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import print_results, test_averaged_run
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
TEST_RUNS = 10
SEED = 42

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_logical_device_configuration(gpu,
    [tf.config.LogicalDeviceConfiguration(memory_limit=2*1024)])


_20news = fetch_20newsgroups(data_home="./data")
y_train = _20news.target
vectorizer = TfidfVectorizer(max_features=2000)
X_train = vectorizer.fit_transform(_20news.data)
X_train = X_train.toarray()


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