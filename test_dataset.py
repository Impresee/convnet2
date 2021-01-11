import tensorflow as tf
import numpy as np


# a = np.array([[[1,1,1],[2,2,2], [3,3,3]],[[4,4,4],[5,5,5],[6,6,6]]])
# print(a.shape)
# print(a)
def parser(x,y):
    r = tf.random.uniform((),0,1)
    print(r)
    return x + r,y
 
 
X = np.array([1,2,3,4,5,6])#np.arange(6).astype(np.float32)
y = X**2
dataset = tf.data.Dataset.from_tensor_slices((X,y))
dataset = dataset.shuffle(6)
dataset = dataset.batch(2)

  
  
# #@tf.function
def log_inputs(inputs):    
    tf.print(inputs)
    return inputs
   
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(log_inputs),
    tf.keras.layers.Dense(1)
])
model.compile(loss = 'mse', optimizer="sgd")
model.fit(dataset, epochs=3, verbose=0)