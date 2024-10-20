import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import  EarlyStopping,ModelCheckpoint
from cleverhans.tf2.utils import optimize_linear

# detect and init the TPU: training on imagenet was done on GCloud TPUs
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)


# instantiate a distribution strategy
tpu_strategy = tf.distribute.TPUStrategy(resolver)

print("Tensorflow version " + tf.__version__)
print("REPLICAS: ", tpu_strategy.num_replicas_in_sync)
print('DEVICES AVAILABLE: {}'.format(tpu_strategy.num_replicas_in_sync))



mapping = {}
images = []

with open('imagenet-object-localization-challenge_2/LOC_synset_mapping.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        mapping[line.split()[0]] = ' '.join(line.split(',')[0].split()[1:])
        
with open('imagenet-object-localization-challenge_2/ILSVRC/ImageSets/CLS-LOC/train_cls.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        images.append(line.split()[0])

basedir = "imagenet-object-localization-challenge_2/ILSVRC/Data/CLS-LOC/train/"
print("reading evalaution")
valdiation= tf.keras.utils.image_dataset_from_directory("evaluation",image_size=(224, 224),batch_size=32 * tpu_strategy.num_replicas_in_sync)
print("reading training")
train_set = tf.keras.utils.image_dataset_from_directory(basedir,image_size=(224, 224),batch_size=32 * tpu_strategy.num_replicas_in_sync)
train_set= train_set.rebatch(32*8, True)

class CustomModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.SparseCategoricalCrossentropy(name="loss")
#         self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.sparscat_metric=tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')
        self.delta = tf.Variable(tf.zeros([32,224,224,3]))
    

    
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

    def fast_gradient_method(self, grad, x, eps, norm, y,):
        # cast to tensor if provided as numpy array
        x = tf.cast(x, tf.float32)

        optimal_perturbation = optimize_linear(grad, eps, norm)
        # Add perturbation to original example to obtain adversarial example
        adv_x = x + optimal_perturbation

        return adv_x
    @property
    def metrics(self):
        # We list our Metric objects here so that reset_states() can be
        # called automatically at the start of each epoch
        # or at the start of evaluate().
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
    densenet=tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=True,
    weights=None,
    input_shape=(224,224 ,3),
    classifier_activation='softmax',
    classes=1000
    )
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Lambda(tf.keras.applications.efficientnet.preprocess_input, input_shape=(224, 224, 3)))
    model.add(densenet)
    
    model = CustomModel(inputs=model.input,outputs=model.output)


    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.1,
        decay_steps=5004*8*4,
        decay_rate=0.1,
        staircase=True)

    model.compile(optimizer=tf.keras.optimizers.experimental.SGD(learning_rate=lr_schedule,momentum=0.9,weight_decay=0.0001),loss = "sparse_categorical_crossentropy")



cb = [EarlyStopping(patience=5, monitor='val_loss', mode='auto' ,restore_best_weights=True,min_delta = 0.01,verbose = True),
      ModelCheckpoint("checkpoint/adv_model",save_best_only=True,monitor = 'val_loss',save_weights_only = 1),PlotLearning()]


history =  model.fit(train_set,validation_data=valdiation,callbacks = cb ,verbose = True, epochs = 90 ,batch_size=32 * tpu_strategy.num_replicas_in_sync)

model.save('saving/adv_model')

