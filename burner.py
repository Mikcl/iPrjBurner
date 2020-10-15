import tensorflow as tf
import numpy as np
import torch

CHANNELS = 4
SQUARE = 10

t = tf.constant([[
    [[i for _ in range(CHANNELS)] for i in range(1,SQUARE+1)] for _ in range(SQUARE)
]])


tT = torch.Tensor([[
    [[i for _ in range(CHANNELS)] for i in range(1,SQUARE+1)] for _ in range(SQUARE)
]])

print('T SHAPE', t.shape)
print('torch shape', tT.shape)

# TODO TODO TODO TODO

# Re adjust code for NCHW
# USE simple net like they did and try it. 

# TODO TODO TODO TODO 





def get_living_mask(x):
  alpha = x[:, :, :, 3:4]
  # input 4d tensor, ksize (size of window for each dimension), stride for each dimension 
  return torch.nn.functional.max_pool2d(input=alpha, kernel_size=3, stride=1, padding=1) > 3
#   return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1

def tf_get_living_mask(x):
  alpha = x[:, :, :, 3:4]
  # input 4d tensor, ksize (size of window for each dimension), stride for each dimension 
  return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 3 #> 0.1

def tf_to_alpha(x):
  return tf.clip_by_value(x[..., 3:4], 3.0, 5.0)

def to_alpha(x):
  return torch.clamp(x[..., 3:4], min=3.0, max=5.0)

# fire_rate = 0.5
# tf_random_mask = tf.random.uniform(tf.shape(t[:, :, :, :1])) <= fire_rate
# random_mask = (torch.FloatTensor(*list(tT[:,:,:,:1].size())).uniform_(0.0, 1.0) <= fire_rate).type(torch.FloatTensor)

# print ('TF MaAK', tf_random_mask)
# print ('mask', random_mask)   
# print('mask shape', random_mask.shape)


def perceive(x, angle=0.0):
    identify = np.float32([0, 1, 0])
    identify = torch.Tensor(np.outer(identify, identify))
    sobel = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter

    dx = torch.Tensor(sobel)
    dy = torch.Tensor(sobel.T)

    c, s = torch.cos(torch.Tensor([angle])), torch.sin(torch.Tensor([angle]))
    x_direction =  c*dx-s*dy
    y_direction =  s*dx+c*dy

    # TODO - Change earlier in code
    x = x.permute(0, 3, 1, 2)

    i_kernel = identify[None, None, ...].repeat(CHANNELS, 1, 1, 1)  # TODO this will always be the same.
    i_v = torch.nn.functional.conv2d(x, i_kernel, padding=1, groups=CHANNELS)

    x_kernel = x_direction[None, None, ...].repeat(CHANNELS, 1, 1, 1)
    x_v = torch.nn.functional.conv2d(x, x_kernel, padding=1, groups=CHANNELS)
    y_kernel = y_direction[None, None, ...].repeat(CHANNELS, 1, 1, 1)
    y_v = torch.nn.functional.conv2d(x, y_kernel, padding=1, groups=CHANNELS)

    stacked_image = torch.cat([i_v, x_v, y_v], 1)
    return stacked_image

    # Kernel - (k):  Height, Width, Channels in image, number of filters
    # (kH, kW, iC, F)
    # Images - (i): number of images in batch, Height, Width, channels, 
    # (N, iH, iW, iC)
    # 
    # Output SHAPE (N, iH, iW, iC×F)

    # NCHW format accepted

    # INPUT , FILTER, STRIDES, PADDING 
    # y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME') # TODO - CHANGE, then try model with simple DenseNet and check error to see if working.
    # KERNEL SHAPE (3, 3, 16, 3)
    # X SHAPE (1, 3, 3, 16)
    # Y SHAPE (1, 3, 3, 48)
    
    # Actual Below
    # KERNEL SHAPE (3, 3, 16, 3)
    # X SHAPE (8, 72, 72, 16)
    # Y SHAPE (8, 72, 72, 48)
perceive(tT)

# resultT = get_living_mask(tT)
# print('RESULT', resultT.shape)
# print('TORCH', resultT)


# result = tf_get_living_mask(t)
# print('RESULT', result.shape)
# print('tf', result)

# print('T', t)

# tT = torch.Tensor([
#     [[i for _ in range(4)] for i in range(1,3)] for _ in range(2)
# ])


# TARGET_PADDING = 1
# CHANNEL_N = 16

# p = TARGET_PADDING
# pad_target = tf.pad(t, [(p, p), (p, p), (0, 0)])

# pad_target_T = torch.nn.functional.pad(tT, pad=(0,0,p,p,p,p), mode='constant', value=0)

# print ('T', t)
# print ('padded', pad_target)
# print ('padded shape', pad_target.shape)
# print ('tT', tT)
# print ('tT shape', tT.shape)
# print ('padded', pad_target_T)
# print ('padded_T shape ', pad_target_T.shape)

# h, w = pad_target.shape[:2]
# seed = np.zeros([h, w, CHANNEL_N], np.float32)
# seed[h//2, w//2, 3:] = 1.0

# def to_rgba(x):
#   return x[..., :4]

# def loss_f(x):
#   return tf.reduce_mean(tf.square(to_rgba(x)-pad_target), [-2, -3, -1])


# loss_log = []

# lr = 2e-3
# lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#     [2000], [lr, lr*0.1])
# trainer = tf.keras.optimizers.Adam(lr_sched)

# loss0 = loss_f(seed).numpy()

# BATCH_SIZE = 8 
# # loss_f(x), where x.shape(N,72,72,16)
# x = tf.constant([
#     [[[i for _ in range(16)] for i in range(72)] for _ in range(72)] for _ in range(BATCH_SIZE)
# ])
# print (x.shape)
# init_loss = loss_f(x)
# print (init_loss)
# loss = tf.reduce_mean(init_loss)
# print (loss)

# # Start initial cell with 0,0,0,1,1,1,1,...,1

# # From percieve Y SHAPE (8, 72, 72, 48)
# #  48 is 3x16 for each pixel.
# # Alive mask will set cells to 0, hence will have no delta.