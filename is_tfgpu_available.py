# Check tf-gpu availability

import tensorflow as tf

if __name__ == "__main__":
    print(f"tf-gpu availability: {tf.test.is_gpu_available()}")