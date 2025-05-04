import numpy as np

# ---------- Helper Functions ----------

def relu_forward(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = dA * (Z > 0)
    return dZ




def dropout_backward(dA_drop, mask, dropout_rate):
    dA = dA_drop * mask
    return dA

import numpy as np

def conv2d_backward(dout, x, filters, stride=1, padding="same"):
    batch_size, in_h, in_w, in_c = x.shape
    out_c, k_h, k_w, _ = filters.shape  # filters are now (out_c, k_h, k_w, in_c)

    if padding == "same":
        pad_h = ((in_h - 1) * stride + k_h - in_h) // 2
        pad_w = ((in_w - 1) * stride + k_w - in_w) // 2
    else:
        pad_h = pad_w = 0

    x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    dx_padded = np.zeros_like(x_padded)
    dfilters = np.zeros_like(filters)
    dbias = np.zeros(out_c)

    out_h, out_w = dout.shape[1], dout.shape[2]

    for b in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                for oc in range(out_c):
                    h_start = i * stride
                    w_start = j * stride

                    # Access filter correctly: filters (out_c, k_h, k_w, in_c)
                    patch = x_padded[b, h_start:h_start + k_h, w_start:w_start + k_w, :]  # (k_h, k_w, in_c)

                    # Update dx_padded and dfilters based on filter shape
                    dx_padded[b, h_start:h_start + k_h, w_start:w_start + k_w, :] += filters[oc] * dout[b, i, j, oc]
                    dfilters[oc] += patch * dout[b, i, j, oc]
                    dbias[oc] += dout[b, i, j, oc]

    # Unpad dx_padded to get dx
    dx = dx_padded[:, pad_h:pad_h + in_h, pad_w:pad_w + in_w, :] if padding == "same" else dx_padded
    return dx, dfilters, dbias



def max_pool(x, size=2, stride=2):
    # Big debug print with an empty row at the start
    # print("\n\n" + "="*50)
    # print(f"Input x shape: {x.shape}")
    # print("="*50 + "\n")
    
    # Extract the shape of the input
    batch, height, width, channels = x.shape
    # print(f"Batch: {batch}, Channels: {channels}, Height: {height}, Width: {width}")

    # Calculate the output dimensions
    out_h = (height - size) // stride + 1
    out_w = (width - size) // stride + 1
    # print(f"Output Height: {out_h}, Output Width: {out_w}")

    # Initialize the output and mask arrays
    out = np.zeros((batch, out_h, out_w, channels))
    mask = np.zeros_like(x)

    # Apply max pooling
    for b in range(batch):
        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    window = x[b,  h_start:h_start+size, w_start:w_start+size,c]
                    max_val = np.max(window)
                    out[b, i, j,c] = max_val
                    mask[b, h_start:h_start+size, w_start:w_start+size,c] += (window == max_val)
    
    # Big debug print with an empty row at the end
    # print("\n\n" + "="*50)
    # print(f"Output shape: {out.shape}")
    # print("="*50 + "\n")
    
    return out, mask


def max_pool_backward(dout, mask, size=2, stride=2):
    dx = np.zeros_like(mask)
    batch, out_h, out_w , channels = dout.shape  
    for b in range(batch):
        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    dx[b,  h_start:h_start+size, w_start:w_start+size,c] += (
                        mask[b,  h_start:h_start+size, w_start:w_start+size,c] * dout[b,  i, j,c]
                    )
    return dx




def batch_norm_backward(dA_out, cache):
    A, mean, var, A_norm, gamma, beta, epsilon = cache
    m = A.shape[0] * A.shape[1] * A.shape[2]  # total elements per channel

    # Gradients of beta and gamma along axes (0, 1, 2)
    dgamma = np.sum(dA_out * A_norm, axis=(0, 1, 2))  # shape: (C,)
    dbeta = np.sum(dA_out, axis=(0, 1, 2))            # shape: (C,)

    dA_norm = dA_out * gamma  # broadcasting over NHWC
    dvar = np.sum(dA_norm * (A - mean) * -0.5 * (var + epsilon) ** (-1.5), axis=(0, 1, 2), keepdims=True)
    dmean = np.sum(dA_norm * -1 / np.sqrt(var + epsilon), axis=(0, 1, 2), keepdims=True) + \
            dvar * np.sum(-2 * (A - mean), axis=(0, 1, 2), keepdims=True) / m
    dA = dA_norm / np.sqrt(var + epsilon) + dvar * 2 * (A - mean) / m + dmean / m

    return dA, dgamma, dbeta






def dropout_forward(x, drop_prob):
    mask = (np.random.rand(*x.shape) > drop_prob).astype(np.float32)
    return x * mask / (1.0 - drop_prob), mask

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)



def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def conv2d(x, filters, bias, stride=1, padding="same"):
    """
    x: shape (batch_size, height, width, in_channels)
    filters: shape (out_channels, kernel_height, kernel_width, in_channels)
    bias: shape (out_channels,)
    Returns: output of shape (batch_size, height, width, out_channels)
    """

    # Extract input shape
    batch_size, in_h, in_w, in_c = x.shape  # input channels are the last dimension
    out_c, k_h, k_w, _ = filters.shape  # output channels, kernel height, kernel width, input channels of filter

    # Debug print for input and filter shapes
    # print(f"Input shape: {x.shape}")
    # print(f"Filter shape: {filters.shape}")
    # print(f"Initial input channels: {in_c}, Output channels: {out_c}, Kernel size: ({k_h}, {k_w})")

    # Padding calculation
    if padding == "same":
        pad_h = (k_h - 1) // 2
        pad_w = (k_w - 1) // 2
    else:
        pad_h = pad_w = 0

    # Debug print for padding values
    # print(f"Padding applied: height={pad_h}, width={pad_w}")

    # Padding the input
    x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")

    # Debug print for padded input shape
    # print(f"Padded input shape: {x_padded.shape}")

    # Output dimension calculation
    out_h = (in_h + 2 * pad_h - k_h) // stride + 1
    out_w = (in_w + 2 * pad_w - k_w) // stride + 1

    # Debug print for output dimensions
    # print(f"Output dimensions (height, width): ({out_h}, {out_w})")

    # Ensure that output dimensions are positive
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Output dimensions are too small: out_h={out_h}, out_w={out_w}.")

    # Initialize output tensor
    out = np.zeros((batch_size, out_h, out_w, out_c))  

    # Debug print for initialized output shape
    # print(f"Initialized output shape: {out.shape}")

    # Convolution operation
    for b in range(batch_size):
        for oc in range(out_c):  # loop over output channels
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride

                    # Perform convolution by summing over all input channels
                    region = x_padded[b, h_start:h_start + k_h, w_start:w_start + k_w, :]

                    # Debug print for the region and its shape
                

                    # Apply the filter to the region (sum over input channels)
                    out[b, i, j, oc] = np.sum(region * filters[oc, :, :, :], axis=(0, 1, 2)) + bias[oc]  # apply filter to the region

                    # Debug print for output value at (b, i, j, oc)

    return out








def maxpool_forward(A, pool_size=2, stride=2):
    H, W, C = A.shape  # TODO check if right
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    A_pool = np.zeros((H_out, W_out, C))
    cache = {}
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size
                A_slice = A[h_start:h_end, w_start:w_end, c]
                A_pool[i, j, c] = np.max(A_slice)
                # Save mask for backpropagation
                mask = (A_slice == np.max(A_slice))
                cache[(i, j, c)] = (mask, (h_start, h_end, w_start, w_end))
    cache["A_prev"] = A  # Save previous layer's output in the cache
    return A_pool, cache


def maxpool_backward(dA, pool_cache, A_shape, pool_size=2, stride=2):
    dA_prev = np.zeros(A_shape)
    H_out, W_out, C = dA.shape  # TODO check if right
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                mask, (h_start, h_end, w_start, w_end) = pool_cache[(i, j, c)]
                dA_prev[h_start:h_end, w_start:w_end, c] += mask * dA[i, j, c]
    return dA_prev

def flatten_forward(A):
    shape = A.shape
    A_flat = A.flatten()
    return A_flat, shape

def flatten_backward(dA_flat, shape):
    return dA_flat.reshape(shape)

def dense_forward(A, W, b):
    Z = np.dot(A, W) + b
    cache = A
    return Z, cache

def dense_backward(dZ, cache, W):
    A = cache  # shape (n,)
    dW = np.outer(A, dZ)
    db = dZ.copy()
    dA = np.dot(dZ, W.T)
    return dA, dW, db

def conv_forward(A_prev, W, b, stride=1, padding="same"):
    # A_prev shape: (H, W, C)
    # W shape: (f, f, C, F)
    # b shape: (F,)
    f = W.shape[0]
    if padding == "same":
        pad = (f // 2) if f % 2 == 0 else (f // 2)
    else:
        pad = 0

    A_pad = np.pad(A_prev, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    H, W_prev, C = A_prev.shape
    H_out = int((H + 2*pad - f) / stride) + 1
    W_out = int((W_prev + 2*pad - f) / stride) + 1
    F = W.shape[3]
    Z = np.zeros((H_out, W_out, F))
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + f
            w_start = j * stride
            w_end = w_start + f
            A_slice = A_pad[h_start:h_end, w_start:w_end, :]
            for k in range(F):
                Z[i, j, k] = np.sum(A_slice * W[:, :, :, k]) + b[k]
    cache = (A_prev, W, b, stride, pad, A_pad)
    return Z, cache

def conv_backward(dZ, cache):
    A_prev, W, b, stride, pad, A_pad = cache
    f = W.shape[0]
    H_out, W_out, F = dZ.shape
    dA_pad = np.zeros_like(A_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + f
            w_start = j * stride
            w_end = w_start + f
            A_slice = A_pad[h_start:h_end, w_start:w_end, :]



            for k in range(F):

               
                
                dW[:, :, :, k] += A_slice * dZ[i, j, k]
                dA_pad[h_start:h_end, w_start:w_end, :] += W[:, :, :, k] * dZ[i, j, k]
                db[k] += dZ[i, j, k]
    if pad != 0:
        dA_prev = dA_pad[pad:-pad, pad:-pad, :]
    else:
        dA_prev = dA_pad
    return dA_prev, dW, db

def batch_norm_forward(A, gamma, beta, epsilon=1e-5):

    # print(f"Shape of A: {A.shape}")
    mean = np.mean(A, axis=(0, 1), keepdims=True)
    var = np.var(A, axis=(0, 1), keepdims=True)
    A_norm = (A - mean) / np.sqrt(var + epsilon)
    A_out = gamma * A_norm + beta
    cache = (A, mean, var, A_norm, gamma, beta, epsilon)
    return A_out, cache




import numpy as np





# ---------- EmotionCNN Class ----------
import numpy as np

class EmotionCNN:
    def __init__(self, filters_list, biases_list, fc_weights, fc_bias, gamma, beta,
                 output_weights, output_bias, step=1, pool_size=2, pool_step=2, 
                 dropout_rate_conv=0.4, dropout_rate_fc=0.3):
        # Initialize all the parameters
        self.filters_list = filters_list
        self.biases_list = biases_list
        self.fc_weights = fc_weights  # List of weights for fully connected layers
        self.fc_bias = fc_bias        # List of biases for fully connected layers
        self.output_weights = output_weights  # Weights for the output layer
        self.output_bias = output_bias        # Bias for the output layer
        self.gamma = gamma            # List of gamma values for batch normalization
        self.beta = beta              # List of beta values for batch normalization
        self.step = step              # Stride for convolution
        self.pool_size = pool_size    # Pooling window size
        self.pool_step = pool_step    # Pooling stride
        self.dropout_rate_conv = dropout_rate_conv  # Dropout rate for conv layers
        self.dropout_rate_fc = dropout_rate_fc      # Dropout rate for fully connected layers

    def forward(self, x, training=True):
        self.cache = {'conv': [], 'bn': [], 'dropout': []}
        out = x

        # --- Conv layers (with ReLU, BN, Pooling, Dropout) ---
        for i in range(4):
            filt = self.filters_list[i]
            bias = self.biases_list[i]
            gamma = self.gamma[i]
            beta = self.beta[i]

            # print(f"\nLayer {i + 1}")
            # print(f"Input shape: {out.shape}")
            # print(f"Filter shape: {filt.shape}")
            # print(f"Bias shape: {bias.shape}")
            # print(f"Gamma shape: {gamma.shape}, Beta shape: {beta.shape}")
            # print(f"Starting a new  conv2d with shape : {out.shape}")
            self.cache['conv_input'] = self.cache.get('conv_input', []) + [out.copy()]

            out = conv2d(out, filt, bias, stride=self.step, padding="same")
            # print(f"After conv2d: {out.shape}")

            out, bn_cache = batch_norm_forward(out, gamma, beta)
            self.cache['bn'].append(bn_cache)

            # Store pre-ReLU activation
            self.cache['pre_relu'] = self.cache.get('pre_relu', []) + [out.copy()]  # Save before ReLU

            out = relu(out)

            # print(f"After ReLU: {out.shape}")

            out, pool_cache = max_pool(out, self.pool_size, self.pool_step)  # TODO shouldn affect no of channels, shpuld have size
            # print(f"After max_pool: {out.shape}")
            self.cache['conv'].append((out, pool_cache))

            if training:
                out, drop_mask = dropout_forward(out, self.dropout_rate_conv)
                # print(f"After dropout (training): {out.shape}")
                self.cache['dropout'].append(drop_mask)
            else:
                # print(f"After dropout (inference): {out.shape}")
                self.cache['dropout'].append(None)


            # Flatten
        self.flatten_shape = out.shape
        # print(f"Flattened shape: {self.flatten_shape}")
        out = out.reshape(out.shape[0], -1)
        self.cache['flat'] = out  # Save the flattened feature map

        # print(f"Shape after flattening: {out.shape}")

        # List of weights and biases for the fully connected layers
        fc_weights = [self.fc_weights[0], self.fc_weights[1]]
        fc_bias = [self.fc_bias[0], self.fc_bias[1]]
        a = out  # Start with the flattened output

        # Iterate over FC layers
        for i in range(2):
            z = a @ fc_weights[i] + fc_bias[i]
            # print(f"Shape after FC layer {i+1} (z): {z.shape}")
            a = relu(z)
            # print(f"Shape after ReLU activation (a): {a.shape}")
            
            # Apply dropout if training
            if training:
                a, drop_mask = dropout_forward(a, self.dropout_rate_fc)
                # print(f"Shape after dropout (a): {a.shape}")
            else:
                drop_mask = None
            
            # Store cache for backpropagation
            self.cache[f'z{i+1}'] = z
            self.cache[f'a{i+1}'] = a
            self.cache[f'drop{i+1}'] = drop_mask

        # --- Output layer ---
        scores = a @ self.output_weights + self.output_bias
        # print(f"Shape after output layer (scores): {scores.shape}")
        probs = softmax(scores)
        # print(f"Shape after softmax (probs): {probs.shape}")

        # Cache for backward
        self.cache.update({
            'scores': scores, 'probs': probs
        })

        return probs



    def backward(self, x, y_true, learning_rate=0.001):
        grads = {}

        # --- Output gradient ---
        m = y_true.shape[0]
        d_scores = self.cache['probs'] - y_true

        # Output layer gradients
        grads['output_w'] = self.cache['a2'].T @ d_scores / m
        grads['output_b'] = np.sum(d_scores, axis=0) / m

        d_a2 = d_scores @ self.output_weights.T

        # Dropout 2 backward
        if self.cache['drop2'] is not None:
            d_a2 *= self.cache['drop2']

        # FC2 backward
        d_z2 = d_a2 * relu_derivative(self.cache['z2'])
        grads['fc_w2'] = self.cache['a1'].T @ d_z2 / m
        grads['fc_b2'] = np.sum(d_z2, axis=0) / m
        d_a1 = d_z2 @ self.fc_weights[1].T

        # Dropout 1 backward
        if self.cache['drop1'] is not None:
            d_a1 *= self.cache['drop1']

        # FC1 backward
        d_z1 = d_a1 * relu_derivative(self.cache['z1'])

        # print("flat.T shape:", self.cache['flat'].T.shape)
        # print("d_z1 shape:", d_z1.shape)


        grads['fc_w1'] = self.cache['flat'].T @ d_z1 / m
        grads['fc_b1'] = np.sum(d_z1, axis=0) / m
        d_flat = d_z1 @ self.fc_weights[0].T
        d_conv_out = d_flat.reshape(self.flatten_shape)

        # --- Conv & Pooling backward ---
        for i in reversed(range(4)):
            # Dropout backward
            drop_mask = self.cache['dropout'][i]
            if drop_mask is not None:
                d_conv_out *= drop_mask

            # Max pool backward
            conv_out, pool_cache = self.cache['conv'][i]
            d_relu = max_pool_backward(d_conv_out, pool_cache)

            # ReLU backward

            pre_relu = self.cache['pre_relu'][i]
            d_bn = d_relu * relu_derivative(pre_relu)


            # Batch norm backward
            d_bn, d_gamma, d_beta = batch_norm_backward(d_bn, self.cache['bn'][i])
            grads[f'gamma_{i}'] = d_gamma
            grads[f'beta_{i}'] = d_beta

            # Conv backward

            conv_input = self.cache['conv_input'][i]
            d_conv_out, dfilt, dbias = conv2d_backward(d_bn, conv_input, self.filters_list[i])

            grads[f'filt_{i}'] = dfilt / m
            grads[f'bias_{i}'] = dbias / m


        # --- Update weights ---
        self.fc_weights[0] -= learning_rate * grads['fc_w1']
        self.fc_bias[0] -= learning_rate * grads['fc_b1']
        self.fc_weights[1] -= learning_rate * grads['fc_w2']
        self.fc_bias[1] -= learning_rate * grads['fc_b2']


        # Update output layer weights and biases
        self.output_weights -= learning_rate * grads['output_w']
        self.output_bias -= learning_rate * grads['output_b']

        for i in range(4):
            self.filters_list[i] -= learning_rate * grads[f'filt_{i}']
            self.biases_list[i] -= learning_rate * grads[f'bias_{i}']
            self.gamma[i] -= learning_rate * grads[f'gamma_{i}']
            self.beta[i] -= learning_rate * grads[f'beta_{i}']
