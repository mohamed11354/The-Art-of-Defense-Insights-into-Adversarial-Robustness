import tensorflow as tf


# detect and init the TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)


# instantiate a distribution strategy
tpu_strategy = tf.distribute.TPUStrategy(resolver)

class evaluateModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
      
        self.clean_loss = tf.keras.metrics.SparseCategoricalCrossentropy(name="cleanloss")
        self.clean_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='cleanacc')
        
        self.pgdloss1 = tf.keras.metrics.SparseCategoricalCrossentropy(name="pgdloss1")
        self.pgdloss2 = tf.keras.metrics.SparseCategoricalCrossentropy(name="pgdloss2")
        self.pgdloss3 = tf.keras.metrics.SparseCategoricalCrossentropy(name="pgdloss3")
        self.pgdloss4 = tf.keras.metrics.SparseCategoricalCrossentropy(name="pgdloss4")
        self.pgdloss5 = tf.keras.metrics.SparseCategoricalCrossentropy(name="pgdloss5")


        self.pgdacc1=tf.keras.metrics.SparseCategoricalAccuracy(name='pgdacc1')
        self.pgdacc2=tf.keras.metrics.SparseCategoricalAccuracy(name='pgdacc2')
        self.pgdacc3=tf.keras.metrics.SparseCategoricalAccuracy(name='pgdacc3')
        self.pgdacc4=tf.keras.metrics.SparseCategoricalAccuracy(name='pgdacc4')
        self.pgdacc5=tf.keras.metrics.SparseCategoricalAccuracy(name='pgdacc5')


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        metrics_list = [ self.clean_acc, self.clean_loss,
            self.pgdloss1, self.pgdloss2, self.pgdloss3, self.pgdloss4, self.pgdloss5,
            self.pgdacc1, self.pgdacc2, self.pgdacc3, self.pgdacc4, self.pgdacc5,
        ]
        return metrics_list
    
    def test_step(self, data):
        # Unpack the data
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(y=y, y_pred=y_pred)
        self.clean_acc.update_state(y, y_pred)
        self.clean_loss.update_state(y,y_pred)
        # Compute predictions
        for eps in range(1,6):
            adv_x=self.projected_gradient_descent(x,eps,eps/20,20,y=y)
            y_pred = self(adv_x, training=False)
            self.compute_loss(y=y, y_pred=y_pred)
            self.metrics[eps+1].update_state(y, y_pred)
            self.metrics[eps+6].update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).        
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def projected_gradient_descent(self,x,eps,eps_iter,nb_iter,y):
        eta = tf.zeros_like(x)
        adv_x = x + eta
        
        i = 0
        while i < nb_iter:
            adv_x = self.fast_gradient_method(adv_x,eps_iter,y=y)

            # Clipping perturbation eta to norm norm ball
            eta = adv_x - x
            eta = tf.clip_by_value(eta, -eps, eps)
            adv_x = x + eta

            i += 1
        return adv_x
    
    @tf.function
    def fast_gradient_method(self,x,eps,y):
        # cast to tensor if provided as numpy array
        x = tf.cast(x, tf.float32)
        
        with tf.GradientTape() as g:
            g.watch(x)
            predictions = self(x,training=False)
            loss=self.compute_loss(y=y,y_pred=predictions)
        grad = g.gradient(loss, x)

        optimal_perturbation = tf.sign(grad)
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
        scaled_perturbation = tf.multiply(eps, optimal_perturbation)

        # Add perturbation to original example to obtain adversarial example
        adv_x = x + scaled_perturbation
        return adv_x
    
# Initalize base model
with tpu_strategy.scope():

    model=tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=True,
        weights='imagenet',
        classifier_activation='softmax',
        classes=1000
    )

    inputs = tf.keras.layers.Input((224, 224, 3))
    preprocessing = tf.keras.layers.Lambda(tf.keras.applications.efficientnet.preprocess_input)(inputs)
    outputs = model(preprocessing)

    base_model= evaluateModel(inputs = inputs, outputs = outputs)

    base_model.compile(optimizer="sgd",loss = "sparse_categorical_crossentropy",metrics=['sparse_categorical_accuracy'])

    # Initlaize new model


    model=tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=True,
    weights=None,
    input_shape=(224,224 ,3),
    classifier_activation='softmax',
    classes=1000
    )
    newmodel = tf.keras.Sequential()
    newmodel.add(tf.keras.layers.Lambda(tf.keras.applications.efficientnet.preprocess_input, input_shape=(224, 224, 3)))
    newmodel.add(model)
    
    newmodel = evaluateModel(inputs=newmodel.input,outputs=newmodel.output)


    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.3,decay_steps=5005,decay_rate=0.01,staircase=True)

    newmodel.compile(optimizer="sgd",loss = "sparse_categorical_crossentropy",metrics=['sparse_categorical_accuracy'])

    newmodel.load_weights("efficientnetb0/checkpoint/adv_densenet")




#Load Data
eval_dir = "evaluation"
# Evaluate New model
eval_set = tf.keras.utils.image_dataset_from_directory(eval_dir,image_size=(224, 224), seed = 4854,batch_size=32*8) 
newmodel.evaluate(eval_set)


# Evaluate base model

eval_set = tf.keras.utils.image_dataset_from_directory(eval_dir,image_size=(224, 224), seed = 4854,batch_size=32*8) # put size of effeicentnet input size
base_model.evaluate(eval_set)



#create summary file
with open('Evaluate.txt','w') as file:
    file.write('Base model:               New model:\n')
    for mb, mn in zip(base_model.metrics,newmodel.metrics):
        file.write(f"{mb.name}: {mb.result():.4f}               {mn.name}: {mn.result():.4f}\n")


