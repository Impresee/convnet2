"""
 author: jsaavedr
 April, 2020 
 This is a general implementation of ResNet
 The implementations should be located in convnet 
"""
import tensorflow as tf


# a conv 3x3
def conv3x3(channels, stride = 1, **kwargs):
    return tf.keras.layers.Conv2D(channels, (3,3), 
                                  strides = stride, 
                                  padding = 'same', 
                                  kernel_initializer = 'he_normal', 
                                  **kwargs)

def conv1x1(channels, stride = 1, **kwargs):
    return tf.keras.layers.Conv2D(channels, 
                                  (1,1), 
                                  strides = stride, 
                                  padding = 'same', 
                                  kernel_initializer = 'he_normal',
                                  **kwargs)

"""
Squeeze and Excitation Block
r_channels is the number of reduced channels
"""
class SEBlock(tf.keras.Model):
    def __init__(self, channels, r_channels, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.gap  = tf.keras.layers.GlobalAveragePooling2D(name = 'se_gap')
        self.fc_1 = tf.keras.layers.Dense(r_channels, name = 'se_fc1' )
        self.bn_1 = tf.keras.layers.BatchNormalization(name = 'se_bn1')        
        self.fc_2 = tf.keras.layers.Dense(channels, name = 'se_fc2')        
                
    def call(self, inputs, training = True):       
        y = self.gap(inputs)
        y = tf.keras.activations.relu(self.bn_1(self.fc_1(y), training))
        scale = tf.keras.activations.sigmoid(self.fc_2(y))
        scale = tf.reshape(scale, (-1,1,1,self.channels))
        y = tf.math.multiply(inputs, scale)
        return y        
        
"""
residual block implementated in a full preactivation mode
input bn-relu-conv1-bn-relu-conv2->y-------------------
  |                                                    |+
  ------------------(projection if necessary)-->shortcut--> y + shortcut
    
""" 
class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, stride, use_projection = False, se_factor = 0,  **kwargs):        
        super(ResidualBlock, self).__init__(**kwargs)
        self.bn_0 = tf.keras.layers.BatchNormalization(name = 'bn_0')
        self.conv_1 = conv3x3(filters, stride, name = 'conv_1')
        self.bn_1 = tf.keras.layers.BatchNormalization(name = 'bn_1')
        self.conv_2 = conv3x3(filters, 1, name = 'conv_2')
        self.use_projection = use_projection;
        self.projection = 0
        if self.use_projection :                            
            self.projection = conv3x3(filters, stride, name = 'projection')
        
        self.se = 0
        self.use_se_block = False
        if se_factor > 0 :
            self.se = SEBlock(filters, filters / se_factor)
            self.use_se_block = True
        
    #using full pre-activation mode    
    def call(self, inputs, training = True):
        y = self.bn_0(inputs)
        y = tf.keras.activations.relu(y)
        if self.use_projection :
            shortcut = self.projection(y)
        else :
            shortcut = inputs
        y = self.conv_1(y)
        y = self.bn_1(y, training)
        y = tf.keras.activations.relu(y)
        y = self.conv_2(y)        
        if self.use_se_block :
            y = self.se(y)        
        y = shortcut + y # residual function        
        return y

"""
BottleneckBlock
expansion rate = x4
"""
class BottleneckBlock(tf.keras.Model):
    def __init__(self, filters, stride, use_projection = False, se_factor = 0, **kwargs):        
        super(BottleneckBlock, self).__init__(**kwargs)
        self.bn_0 = tf.keras.layers.BatchNormalization(name = 'bn_0')
        #conv_0 is the compression layer
        self.conv_0 = conv1x1(filters, stride, name = 'conv_0')
        self.conv_1 = conv3x3(filters, 1, name = 'conv_1')
        self.bn_1 = tf.keras.layers.BatchNormalization(name = 'bn_1')
        self.conv_2 = conv1x1(filters * 4, 1, name = 'conv_2')
        self.bn_2 = tf.keras.layers.BatchNormalization(name = 'bn_2')
        self.use_projection = use_projection
        self.projection = 0
        if self.use_projection :                            
            self.projection = conv3x3(filters * 4, stride, name = 'projection')
        self.se = 0
        self.use_se_block = False
        if se_factor > 0 :
            self.se = SEBlock(filters, filters / se_factor)
            self.use_se_block = True
        
    #using full pre-activation mode     
    def call(self, inputs, training = True):
        y = self.bn_0(inputs, training)
        y = tf.keras.activations.relu(y)
        if self.use_projection :
            shortcut = self.projection(y)
        else :
            shortcut = inputs            
        y = self.conv_0(y)
        y = self.bn_1(y, training)
        y = tf.keras.activations.relu(y)
        y = self.conv_1(y)
        y = self.bn_2(y, training)
        y = tf.keras.activations.relu(y)
        y = self.conv_2(y)        
        if self.use_se_block :
            y = self.se(y)        
        y = shortcut + y # residual function        
        return y
"""
    resnet block implementation
    A resnet block contains a set of residual blocks
    Commonly, the residual block of a resnet block starts with a stride = 2, except for the first block
    The number of blocks together with the number of channels are inputs
    
"""
class ResNetBlock(tf.keras.Model):
    def __init__(self, filters,  block_size, with_reduction = False, use_bottleneck = False, se_factor = 0, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)        
        self.filters = filters    
        self.block_size = block_size
        if use_bottleneck :
            residual_block = BottleneckBlock
        else:
            residual_block = ResidualBlock
        #the first block is nos affected by a spatial reduction 
        stride_0 = 1
        use_projection_at_first = False
        if with_reduction :
            stride_0 = 2
            use_projection_at_first = True
        self.block_collector = [residual_block(filters = filters, stride = stride_0, use_projection = use_projection_at_first, se_factor = se_factor, name = 'rblock_0')]
        for idx_block in range(1, block_size) :
            self.block_collector.append(residual_block(filters = filters, stride = 1, se_factor = se_factor, name = 'rblock_{}'.format(idx_block)))
                
    def call(self, inputs, training):
        x = inputs;
        for block in self.block_collector :
            x = block(x, training)
        return x;

""" 
 ResNet model 
 e.g.    
 block_sizes: it is the number of residual components for each block e.g  [3,3,3] for 3 blocks 
 channels : it is the number of channels within each block [32,64,128]
 se_factor : reduction factor in  SE module, 0 if SE is not used
"""
class ResNet(tf.keras.Model):        
    def __init__(self, block_sizes, filters, number_of_classes, use_bottleneck = False, se_factor = 0, **kwargs) :
        super(ResNet, self).__init__(**kwargs)
        self.conv_0 = tf.keras.layers.Conv2D(64, (7,7), strides = 2, padding = 'same', 
                                             kernel_initializer = 'he_normal', 
                                             name = 'conv_0')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2, padding = 'same')
        self.resnet_blocks = [ResNetBlock(filters = filters[0], 
                                          block_size = block_sizes[0], 
                                          with_reduction = False,  
                                          use_bottleneck = use_bottleneck, 
                                          se_factor = se_factor, 
                                          name = 'block_0')] 
        for idx_block in range(1, len(block_sizes)) :                     
            self.resnet_blocks.append(ResNetBlock(filters = filters[idx_block], 
                                                  block_size = block_sizes[idx_block], 
                                                  with_reduction = True,  
                                                  use_bottleneck = use_bottleneck,
                                                  se_factor = se_factor,
                                                  name = 'block_{}'.format(idx_block)))
        self.bn_last= tf.keras.layers.BatchNormalization(name = 'bn_last')                            
        self.avg_pool = tf.keras.layers.AveragePooling2D()
        #self.fn = tf.keras.layers.Dense(1024)
        #self.bn_fn= tf.keras.layers.BatchNormalization()        
        self.classifier = tf.keras.layers.Dense(number_of_classes)
        
    def call(self, inputs, training):
        x = inputs
        x = self.conv_0(x)
        x = self.max_pool(x)                 
        for block in self.resnet_blocks :
            x = block(x, training)
        x = self.bn_last(x)
        x = tf.keras.activations.relu(x)    
        x = self.avg_pool(x)        
        x = tf.keras.layers.Flatten()(x)        
        x = self.classifier(x)
        return x    
"""
A simple main for doing small experiments
"""    
if __name__ == '__main__' :
    #a = tf.constant([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]], dtype = tf.float32)    
    #b = tf.constant([1,2,3], dtype = tf.float32)
    #b = tf.reshape(b, [1,1,3])
    #print(a)
    #print(b)
    #c = tf.math.multiply(a,b)
    #sess = tf.Session()
    #print(sess.run(c))
    model = ResNet(block_sizes=[3,4,6,3], filters = [16, 128, 256, 512], number_of_classes = 10)
    model.build((1, 224,224,3))
    model.summary()    
    model.save('the-model.pb', save_format='tf')
    #model.save("the-model")
    