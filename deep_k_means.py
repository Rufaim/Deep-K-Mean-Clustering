import tensorflow as tf
import numpy as np
from autoencoder import AutoEncoder
from sklearn.cluster import KMeans

class DeepKMeans(tf.keras.Model):
    def __init__(self,autoencoder,k,alpha=1.0,seed=None):
        # assert isinstance(autoencoder,AutoEncoder)
        super(DeepKMeans,self).__init__()
        self.autoencoder = autoencoder
        self.k = k
        self.alpha = tf.constant(alpha,dtype=tf.float32)
        self.seed = seed
        self.dist = self._euclid_distance

    def build(self, input_shape):
        init = tf.random_uniform_initializer(minval=-1, maxval=1, seed=self.seed)
        self._centers = self.add_weight("centers",
                                        shape=[self.k, int(self.autoencoder.embedding_size)],
                                        initializer=init,
                                        trainable=True)
        self.built = True

    @tf.function
    def _euclid_distance(self,input):
        dists = tf.reduce_sum((tf.expand_dims(input, axis=1) - self._centers) ** 2, axis=-1)
        return dists

    @tf.function
    def _closeness(self,input,alpha):
        dists = self.dist(input)
        alp = tf.cast(alpha,tf.float32)
        qs = 1+dists/alp
        qs = qs**(-(alp+1)/2)
        qs /= tf.reduce_sum(qs,axis=1,keepdims=True)
        # qs = tf.nn.softmax(-alp*dists,axis=-1)
        return dists, qs

    def call(self, inputs ,training=None):
        # rep = self.autoencoder.encode(inputs,training)
        rep = self.autoencoder.encoder(inputs, training)
        dists,qs= self._closeness(rep,self.alpha)
        return tf.argmax(qs,axis=-1), dists

    def fit(self,X,batch_size,pretrain_epochs=300,finetune_epoch=100,update_epoch=10,pretrain_learning_rate=1e-3,learning_rate=1e-3,seed=None,verbose=False):
        seed_gen = np.random.RandomState(seed)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=pretrain_learning_rate)
        data = tf.data.Dataset.range(X.shape[0])

        batch_generator = data.batch(batch_size)

        # ~~~ warmup ~~~
        warmup_input = tf.zeros((1,) + X.shape[1:], dtype=tf.float32)
        self.autoencoder(warmup_input)
        self(warmup_input)
        # ~~~ end of warmup ~~~

        for epoch in range(pretrain_epochs):
            epoch_loss = 0
            batches = 0
            current_seed = seed_gen.randint(0, 10000000)
            for idx in batch_generator.shuffle(10*batch_size,seed=current_seed):
                batch_x = tf.convert_to_tensor(X[idx], dtype=tf.float32)

                with tf.GradientTape() as g:
                    x_hat, rep = self.autoencoder(batch_x)
                    # MSE(x,y) <=> tf.reduce_mean(tf.reduce_mean((batch_x-x_hat) ** 2, axis=1)))
                    loss_ae = tf.reduce_mean(tf.reduce_sum((batch_x-x_hat) ** 2, axis=1))# tf.reduce_mean(tf.keras.losses.MSE(batch_x,x_hat))
                grad = g.gradient(loss_ae, self.autoencoder.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.autoencoder.trainable_variables))
                batches +=1
                epoch_loss += loss_ae.numpy()
            if verbose:
                tf.summary.scalar('Pretrain autoencoder loss', data=epoch_loss/batches, step=epoch)
                print(f"\rPretrain Epoch: # {epoch} loss: {epoch_loss/batches}", end="")

        init_repr = self.autoencoder.encode(X, False)
        init_repr = init_repr.numpy()
        kmean = KMeans(self.k)
        kmean.fit(init_repr)
        self._centers.assign(kmean.cluster_centers_)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        for epoch in range(finetune_epoch):
            epoch_cls_loss = 0

            if epoch % update_epoch == 0:
                rep = self.autoencoder.encode(X, False)
                _, qs = self._closeness(rep, self.alpha * 1000)
                targets = qs**2 / tf.reduce_sum(qs,axis=0,keepdims=True)
                targets /= tf.reduce_sum(targets, axis=1, keepdims=True)

            current_seed = seed_gen.randint(0, 10000000)
            for idx in batch_generator.shuffle(10*batch_size,seed=current_seed):
                batch_x = tf.convert_to_tensor(X[idx],dtype=tf.float32)
                batch_targets = tf.gather(targets, idx, axis=0, batch_dims=0)

                with tf.GradientTape() as g:
                    rep = self.autoencoder._encode(batch_x,True)
                    dists, qs = self._closeness(rep, self.alpha * 1000)
                    loss_cls = tf.reduce_mean(tf.keras.losses.KLD(batch_targets,qs))
                vars = self.autoencoder.encoder.trainable_variables + [self._centers]
                grad = g.gradient(loss_cls, vars)
                self.optimizer.apply_gradients(zip(grad, vars))

                epoch_cls_loss += loss_cls.numpy()
            if verbose:
                tf.summary.scalar('Cluster loss', data=epoch_cls_loss, step=epoch)
                for k in range(self.k):
                    tf.summary.histogram(
                        f"center_{k}", self._centers[k].numpy(), step=epoch, buckets=self.autoencoder.embedding_size, description=None)
                print(f"\rEpoch: # {epoch}", end="")

        if verbose:
            print()
