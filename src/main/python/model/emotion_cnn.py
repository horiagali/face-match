import numpy as np

class EmotionCNN:
    def __init__(self, filters_list, biases_list, fc_weights, fc_bias, step=1, pool_size=2, pool_step=2):
        self.filters_list = filters_list      # List of convolution filters (one per layer)
        self.biases_list = biases_list        # List of biases for each conv layer
        self.fc_weights = fc_weights          # Fully connected layer weights
        self.fc_bias = fc_bias                # Fully connected layer bias
        self.step = step                      # Stride for convolution
        self.pool_size = pool_size            # Pooling window size
        self.pool_step = pool_step            # Stride for pooling

    # --------- Forward functions with caching ----------
    def conv_forward(self, A_prev, W, b):
        """
        A_prev: input image (H, W) - assumes grayscale, no channel dimension
        W: filter (f, f)
        b: scalar bias
        Returns:
          Z: convolution output
          cache: tuple of (A_prev, W, b, stride)
        """
        A_prev = A_prev.astype(np.float64)  # Cast to float64
        H_prev, W_prev = A_prev.shape  # A_prev is now 2D (height, width)
        f = W.shape[0]  # filter size
        H_out = int((H_prev - f) / self.step) + 1
        W_out = int((W_prev - f) / self.step) + 1

        Z = np.zeros((H_out, W_out), dtype=np.float64)  # Ensure Z is float64
        
        for i in range(H_out):
            for j in range(W_out):
                vert_start = i * self.step
                vert_end = vert_start + f
                horiz_start = j * self.step
                horiz_end = horiz_start + f
                A_slice = A_prev[vert_start:vert_end, horiz_start:horiz_end]
                Z[i, j] = np.sum(A_slice * W) + b
        cache = (A_prev, W, b, self.step)
        return Z, cache

    def relu_forward(self, Z):
        A = np.maximum(0, Z)
        cache = Z  # we need Z for backprop
        return A, cache

    def pool_forward(self, A_prev, pool_size, stride):
        A_prev = A_prev.astype(np.float64)  # Cast to float64
        H_prev, W_prev = A_prev.shape
        H_out = int((H_prev - pool_size) / stride) + 1
        W_out = int((W_prev - pool_size) / stride) + 1
        A = np.zeros((H_out, W_out), dtype=np.float64)  # Ensure A is float64
        cache = {"A_prev": A_prev, "pool_size": pool_size, "stride": stride}
        for i in range(H_out):
            for j in range(W_out):
                vert_start = i * stride
                vert_end = vert_start + pool_size
                horiz_start = j * stride
                horiz_end = horiz_start + pool_size
                A_slice = A_prev[vert_start:vert_end, horiz_start:horiz_end]
                A[i, j] = np.max(A_slice)
        return A, cache

    def fc_forward(self, A_flat):
        Z = np.dot(A_flat, self.fc_weights) + self.fc_bias
        cache = A_flat  # cache the flattened input
        return Z, cache

    def forward(self, image):
        image = image.astype(np.float64)  # Ensure input image is float64
        self.cache = {}  # dictionary to hold caches for each layer
        A = image
        self.cache["A0"] = A
        L = len(self.filters_list)
        for l in range(L):
            # Convolution forward
            Z_conv, cache_conv = self.conv_forward(A, self.filters_list[l], self.biases_list[l])
            self.cache[f"conv_{l}"] = cache_conv
            # ReLU forward
            A_relu, cache_relu = self.relu_forward(Z_conv)
            self.cache[f"relu_{l}"] = cache_relu
            # Pooling forward
            A_pool, cache_pool = self.pool_forward(A_relu, self.pool_size, self.pool_step)
            self.cache[f"pool_{l}"] = cache_pool
            A = A_pool  # output becomes input for next layer

        self.cache["A_final"] = A  # final conv block output
        # Flatten
        A_flat = A.flatten()
        self.cache["A_flat"] = A_flat
        # Fully connected forward
        Z_fc, cache_fc = self.fc_forward(A_flat)
        self.cache["fc"] = cache_fc
        self.cache["Z_fc"] = Z_fc
        # Softmax
        probabilities = self.softmax(Z_fc)
        return probabilities

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def cross_entropy_loss(self, predictions, labels):
        epsilon = 1e-15  # Small constant to prevent log(0)
        predictions = np.clip(predictions, epsilon, 1. - epsilon)  # Clip predictions to avoid 0 or 1 values
        m = labels.shape[0]
        loss = -np.sum(labels * np.log(predictions)) / m
        return loss


    # --------- Backward functions for each layer ----------
    def conv_backward(self, dZ, cache):
        A_prev, W, b, stride = cache
        A_prev = A_prev.astype(np.float64)  # Cast A_prev to float64
        W = W.astype(np.float64)            # Cast W to float64
        f = W.shape[0]
        H_prev, W_prev = A_prev.shape
        H_out, W_out = dZ.shape

        dA_prev = np.zeros_like(A_prev, dtype=np.float64)  # Ensure dA_prev is float64
        dW = np.zeros_like(W, dtype=np.float64)            # Ensure dW is float64
        db = 0.0                                           # Bias gradient should be a float

        for i in range(H_out):
            for j in range(W_out):
                vert_start = i * stride
                vert_end = vert_start + f
                horiz_start = j * stride
                horiz_end = horiz_start + f
                A_slice = A_prev[vert_start:vert_end, horiz_start:horiz_end]
                dW += dZ[i, j] * A_slice
                dA_prev[vert_start:vert_end, horiz_start:horiz_end] += dZ[i, j] * W
                db += dZ[i, j]
        return dA_prev, dW, db

    def relu_backward(self, dA, cache):
        Z = cache
        dZ = dA * (Z > 0)
        return dZ

    def pool_backward(self, dA, cache):
        A_prev = cache["A_prev"]
        pool_size = cache["pool_size"]
        stride = cache["stride"]
        H_prev, W_prev = A_prev.shape
        H_out, W_out = dA.shape
        dA_prev = np.zeros_like(A_prev, dtype=np.float64)

        for i in range(H_out):
            for j in range(W_out):
                vert_start = i * stride
                vert_end = vert_start + pool_size
                horiz_start = j * stride
                horiz_end = horiz_start + pool_size
                A_slice = A_prev[vert_start:vert_end, horiz_start:horiz_end]
                mask = (A_slice == np.max(A_slice))
                dA_prev[vert_start:vert_end, horiz_start:horiz_end] += mask * dA[i, j]
        return dA_prev

    def fc_backward(self, dZ, cache):
        A_flat = cache  # flattened input
        dW = np.outer(A_flat, dZ).astype(np.float64)  # Ensure dW is float64
        db = dZ
        dA_flat = np.dot(self.fc_weights, dZ)
        return dA_flat, dW, db

    # --------- Full backpropagation through the network ----------
    def backpropagate(self, image, true_label, learning_rate=0.01):
        # Forward pass (and cache all intermediate values)
        probabilities = self.forward(image)
        # One-hot encode true label (assumes number of classes is same as length of fc_bias)
        num_classes = self.fc_bias.shape[0]
        one_hot_label = np.zeros(num_classes)
        one_hot_label[true_label] = 1

        # Compute loss (for monitoring)
        loss = self.cross_entropy_loss(probabilities, one_hot_label)

        # Backprop through softmax and fully connected layer:
        # For softmax with cross-entropy, gradient at fc output:
        dZ_fc = probabilities - one_hot_label  # shape: (num_classes,)
        dA_flat, dW_fc, db_fc = self.fc_backward(dZ_fc, self.cache["fc"])
        # Update fully connected parameters:
        self.fc_weights -= learning_rate * dW_fc
        self.fc_bias -= learning_rate * db_fc

        # Reshape dA_flat to match the shape of the final conv output:
        dA = dA_flat.reshape(self.cache["A_final"].shape)

        # Backpropagate through convolution blocks in reverse order:
        L = len(self.filters_list)
        for l in reversed(range(L)):
            # Pooling backward:
            dA = self.pool_backward(dA, self.cache[f"pool_{l}"])
            # ReLU backward:
            dZ = self.relu_backward(dA, self.cache[f"relu_{l}"])
            # Convolution backward:
            dA, dW, db = self.conv_backward(dZ, self.cache[f"conv_{l}"])
            # Update convolution parameters:
            self.filters_list[l] -= learning_rate * dW
            self.biases_list[l] -= learning_rate * db

        return loss
