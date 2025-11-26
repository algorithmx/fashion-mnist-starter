import os
import shutil
import subprocess
import matplotlib.pyplot as plt
from dataset_loader import load_numpy

# GPU/CPU selector for TensorFlow (similar idea to PyTorch selector):
# - If an NVIDIA GPU is present but `ptxas` (CUDA assembler) is missing, disable CUDA
#   to avoid PTX compilation warnings and fallback behavior. Users can override by
#   setting the env var `FORCE_TF_GPU=1` to force GPU even if `ptxas` is missing.
# - Respect an existing `CUDA_VISIBLE_DEVICES` setting (empty string disables GPUs).
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Hide TF INFO logs by default

force_gpu = os.environ.get('FORCE_TF_GPU', '0') == '1'
gpu_disabled = False

if os.environ.get('CUDA_VISIBLE_DEVICES', None) == '':
    # User has explicitly disabled GPUs; respect that
    gpu_disabled = True
else:
    try:
        has_nvidia_smi = shutil.which('nvidia-smi') is not None
        has_ptxas = shutil.which('ptxas') is not None
        gpu_unsupported = False
        major = minor = 0
        # Prefer disabling CUDA for older GPUs (compute capability < 7.0)
        if has_nvidia_smi:
            try:
                out = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'
                ], text=True, stderr=subprocess.DEVNULL).strip().splitlines()
                if out:
                    first = out[0].strip()
                    parts = first.split('.')
                    major = int(parts[0]) if parts[0].isdigit() else 0
                    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    if major < 7:
                        gpu_unsupported = True
            except Exception:
                # If query fails, don't assume unsupported
                pass

        # Disable CUDA if ptxas missing or GPU compute capability is unsupported
        if (has_nvidia_smi and (not has_ptxas or gpu_unsupported)) and not force_gpu:
            reason = 'ptxas not found' if not has_ptxas else f'GPU compute capability {major}.{minor} < 7.0'
            print(f"{reason}; disabling CUDA to avoid PTX compilation warnings. Set FORCE_TF_GPU=1 to override.")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            gpu_disabled = True
    except Exception:
        # If any check fails, don't change CUDA visibility (safe default)
        pass

# Now import TensorFlow after CUDA visibility is set
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configure TF device visibility/memory use based on selector outcome
try:
    if gpu_disabled:
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for g in gpus:
                try:
                    tf.config.experimental.set_memory_growth(g, True)
                except Exception:
                    # ignore failures (may be unsupported on some builds)
                    pass
except Exception:
    pass

# Step 1: Load the Fashion-MNIST dataset (prefer local IDX files)
# `load_numpy` will fall back to `keras.datasets.fashion_mnist.load_data()` if needed.
print('Loading Fashion-MNIST via shared dataset_loader (normalized)')
# load_numpy now returns float32 images normalized to [-1,1] and int64 labels
x_train, y_train, x_test, y_test = load_numpy()

# Add channel dimension for TensorFlow: (N, 28, 28) -> (N, 28, 28, 1)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

# Step 2: Define neural network model
class MinimalFashionCNN(keras.Model):
    def __init__(self):
        super(MinimalFashionCNN, self).__init__()
        self.conv1 = layers.Conv2D(16, kernel_size=3, padding='valid', activation=None)
        self.relu = layers.ReLU()
        self.pool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(10)  # Logits output (no activation)

    def call(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = MinimalFashionCNN()

# Step 3: Define loss function and optimizer
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)


# Step 4: Train the model
def train_model(num_epochs=10):
    train_acc_history, val_acc_history = [], []

    for epoch in range(num_epochs):
        # Training
        t_loss, t_corr, n = 0.0, 0, 0
        for x, y in train_dataset:
            with tf.GradientTape() as tape:
                outputs = model(x, training=True)
                loss = loss_fn(y, outputs)
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            t_loss += loss.numpy()
            t_corr += tf.reduce_sum(tf.cast(tf.argmax(outputs, axis=1) == y, tf.int32)).numpy()
            n += len(y)
        
        # Validation
        v_corr, m = 0, 0
        for x, y in test_dataset:
            outputs = model(x, training=False)
            v_corr += tf.reduce_sum(tf.cast(tf.argmax(outputs, axis=1) == y, tf.int32)).numpy()
            m += len(y)
        
        t_acc, v_acc = t_corr / n, v_corr / m
        train_acc_history.append(t_acc)
        val_acc_history.append(v_acc)
        print(f'Epoch {epoch+1}: Loss: {t_loss / len(train_dataset):.4f}, Train Acc: {t_acc:.4f}, Val Acc: {v_acc:.4f}')

    print(f'Test accuracy: {v_acc:.4f}')

if __name__ == '__main__':
    train_model()