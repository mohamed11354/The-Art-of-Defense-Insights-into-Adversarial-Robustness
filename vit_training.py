import tensorflow as tf
import pickle
import Vit
from tensorflow.keras.callbacks import  EarlyStopping,ModelCheckpoint

tf.profiler.experimental.server.start(6000)


# detect and init the TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)


# instantiate a distribution strategy
tpu_strategy = tf.distribute.TPUStrategy(resolver)

print("Tensorflow version " + tf.__version__)
print("REPLICAS: ", tpu_strategy.num_replicas_in_sync)


def get_inception_crop(size=None, area_min=8, area_max=100,
                       method="bilinear", antialias=False):
    
    def _inception_crop(image_label,seed):  # pylint: disable=missing-docstring
        image,label = image_label
        new_seed = tf.random.split(seed, num=1)[0, :]
        begin, crop_size, _ = tf.image.stateless_sample_distorted_bounding_box(
            tf.shape(image),
            tf.zeros([0, 0, 4], tf.float32),
            seed,
            aspect_ratio_range=[0.75,1.333],
            area_range=(area_min / 100, area_max / 100),
            min_object_covered=0,  # Don't enforce a minimum area.
            use_image_if_no_bounding_boxes=True)
        crop = tf.slice(image, begin, crop_size)
        # Unfortunately, the above operation loses the depth-dimension. So we need
        # to restore it the manual way.
        crop.set_shape([None, None, image.shape[-1]])
        crop = tf.image.resize(crop, (size,size), method=method, antialias=antialias)
        #crop = tf.cast(tf.clip_by_value(crop, tf_dtype.min, tf_dtype.max), dtype)
        crop = tf.image.stateless_random_flip_left_right(crop,new_seed)
        return crop, label

    return _inception_crop

inception = get_inception_crop(224)


basedir = "imagenet-object-localization-challenge_2/ILSVRC/Data/CLS-LOC/train/"
eval_dir="evaluation"

print("reading evalaution")
validation= tf.keras.utils.image_dataset_from_directory(eval_dir,image_size=(224, 224),batch_size=32*8).map(lambda x,y:(tf.cast(x, tf.bfloat16),y)) # put size of effeicentnet input size

print("reading training")
train_set = tf.keras.utils.image_dataset_from_directory(basedir,image_size=(224, 224),batch_size=None).map(lambda x,y:(tf.cast(x, tf.bfloat16),y)) # put size of effeicentnet input size

 

counter = tf.data.Dataset.counter()
train_set = tf.data.Dataset.zip((train_set, (counter,counter) ))
train_set = train_set.map(inception)
train_set = train_set.batch(64*4,True)


class CustomModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.SparseCategoricalCrossentropy(name="loss")
        self.sparscat_metric=tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')
        self.delta = tf.Variable(tf.zeros([64,224,224,3]))
    

    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to fit().
        x, y = data
        print('.')
        true_x=x
        eps=2.0
        nb_iter=4.0
        eps_iter=0.6
        
        i = 0
        result={}
        while i < nb_iter:
            cur_noise = self.delta
            with tf.GradientTape() as tape:
                tape.watch(cur_noise)
                x = true_x + cur_noise
                x = tf.clip_by_value(x, 0.0, 255.0)
                predictions = self(x,training=True)
                loss = self.compute_loss(y=y,y_pred=predictions)

                
                
            gradients,adv_grad = tape.gradient(loss, [self.trainable_variables,cur_noise])    
    
            # Clipping perturbation eta to norm norm ball
            holder = tf.sign(adv_grad)*eps_iter
            holder = self.delta+holder

            holder = tf.clip_by_value(holder, -eps, eps)
            self.delta.assign(holder)

            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            for metric in self.metrics:
                metric.update_state(y, predictions)
            # Return a dict mapping metric names to current value
            result= {m.name: m.result() for m in self.metrics}

            i += 1
        return result

    @property
    def metrics(self):
        # We list our Metric objects here so that reset_states() can be
        # called automatically at the start of each epoch
        # or at the start of evaluate().
        # If you don't implement this property, you have to call
        # reset_states() yourself at the time of your choosing.
        return [ self.loss_tracker,self.sparscat_metric]
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compute_loss(y=y, y_pred=y_pred)
        # Update the metrics.
        for metric in self.metrics:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    
class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        self.val_iter=1
        for metric in logs:
            self.metrics[metric] = []

    def on_train_batch_end(self, batch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        if(batch%200==0):
            with open('my_list.pkl', 'wb') as file:
                pickle.dump(self.metrics, file)
    def on_test_end(self,logs={}):
        for metric in logs:
            val_metric='val_'+metric
            if val_metric in self.metrics:
                self.metrics[val_metric].append(logs.get(metric))
            else:
                self.metrics[val_metric] = [logs.get(metric)]

with tpu_strategy.scope():
    #Load baseline Model
    base_model = Vit.ViTB16(
        include_rescaling=True,
        include_top=True,
        num_classes=1000,
        classifier_activation='softmax',
        input_shape=(224,224,3))
    base_model = CustomModel(inputs=base_model.input,outputs=base_model.output)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(0.001,5004*270,warmup_target=0.001,warmup_steps=5004*30)



    base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule,weight_decay=0.1,global_clipnorm=1.0),loss = "sparse_categorical_crossentropy",metrics=['acc'])

cb = [EarlyStopping(patience=5, monitor='val_loss', mode='auto' ,restore_best_weights=True,min_delta = 0.01,verbose = True),
      ModelCheckpoint("checkpoint/adv",save_best_only=True,monitor = 'val_loss',save_weights_only = 1),PlotLearning()]

base_model.fit(train_set,validation_data=validation,callbacks = cb ,verbose = True, epochs = 300 ,batch_size=32 * tpu_strategy.num_replicas_in_sync)
base_model.save('saving/adv')
