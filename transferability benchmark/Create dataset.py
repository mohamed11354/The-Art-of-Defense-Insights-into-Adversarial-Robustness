import os 
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
    

@tf.function
def IG(model, images, labels, steps = 15):
    # interpolated intensities
    alphas = tf.linspace(0.0,1.0,steps+1)
    alphas_x = alphas[:,tf.newaxis, tf.newaxis, tf.newaxis,tf.newaxis]
    alphas_x = tf.reshape(alphas_x,(1,alphas.shape[0],1,1,1))

    # standardize input
    images = tf.cast(images,'float32')

    # calculate interpolated images
    delta = tf.expand_dims(images,1)
    interpolated = delta * alphas_x
    # calculate gradient for each image
    grads = tf.TensorArray(tf.float32,1)
    i = 0
    for inter in interpolated:
        with tf.GradientTape() as tape:
            tape.watch(inter)
            probs = model(inter)[:,labels[0]]
        grads = grads.write(i,tape.gradient(probs,inter))
        
        i+= 1

    grads = grads.stack()

    # aproximate integration using remain trapoziod
    grads = (grads[:,:-1] + grads[:,1:]) / tf.constant(2.0)
    avg_gradients = tf.math.reduce_mean(grads, axis=1)

    integrated_gradient = images * avg_gradients

    return integrated_gradient

@tf.function
def MIG(model, images, labels, epsilon = 16, iterations = 20):
    g_t = tf.zeros_like(images)
    x_t = images
    eps_step = tf.constant(epsilon/iterations)
    for i in tf.range(iterations):
        # steps = 20 as research
        #labels = tf.argmax(model(x_t) , -1)
        delta_t = IG(model,x_t,labels) 
        g_t = g_t + (delta_t/tf.norm(delta_t,1))
        x_h = x_t - eps_step * tf.sign(g_t)
        x_t = tf.clip_by_value(x_h,x_h-epsilon,x_h+epsilon)
    return tf.clip_by_value(x_t,0,255)


# surrogate model you can change as you need
base_model=tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=True,
    weights= None,
    classifier_activation='softmax',
    classes=1000
)

inputs = tf.keras.layers.Input((224, 224, 3))
preprocessing = tf.keras.layers.Lambda(tf.keras.applications.efficientnet.preprocess_input)(inputs)
outputs = base_model(preprocessing)

efficientnetb0 = tf.keras.Model(inputs = inputs, outputs = outputs)
efficientnetb0.compile(optimizer="sgd",loss = "sparse_categorical_crossentropy",metrics=['sparse_categorical_accuracy'])
efficientnetb0.load_weights('/kaggle/input/efficientnetb0-adv-pgd/checkpoint.h5') # path of weights the data generated on the kaggle


basesrc = '/kaggle/input/eval-2k/subeval' # clean dataset path
dirs = sorted(os.listdir(basesrc)) # ensure the ordering of the folders is the same as clean dataset


# adjust how many classes you need to cover
st = 0
end = 1000

basedist = f"/kaggle/working/efficientnetb0_trained_pgd_{end}" # new dataset directory
os.mkdir(basedist)

for label , dirc in zip(range(st,end),dirs[st:end]):
    os.mkdir(basedist+f"/{dirc}")
    subdirs = os.listdir(basesrc+f"/{dirc}")
    for i in range(2):
        img = tf.keras.utils.load_img(basesrc + f"/{dirc}/{subdirs[i]}",target_size = (224,224),interpolation='bicubic')
        img = tf.convert_to_tensor(img,dtype = tf.float32)
        adv_img = MIG(efficientnetb0,tf.expand_dims(img,0),[label])
        plt.imsave(basedist + f"/{dirc}/{subdirs[i]}", np.array(adv_img[0]/255))

# sanity check: accuracy of generated images
evalattacked_dir = f"/kaggle/working/efficientnetb0_trained_pgd_{end}"
evalattacked_set = tf.keras.utils.image_dataset_from_directory(evalattacked_dir,image_size=(224, 224))
efficientnetb0.evaluate(evalattacked_set)