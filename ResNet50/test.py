import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, Adam

#set up parameters
target_side_len = 224
batch_size = 16
epochs = 3
optimizer_choice = Adam
learning_rate = 0.0001

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


# define f1 score
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def f1(y_true, y_pred): 
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2*((precision_value * recall_value)/(precision_value + recall_value + K.epsilon()))

# Load the model with custom objects (including the custom metric)
loaded_model = tf.keras.models.load_model('ResNet50_topWithBaseLine_model.h5', custom_objects={'f1': f1})

# Load the weights into the model
loaded_model.load_weights('ResNet50_topWithBaseLine_weights.h5')

# Define a function to preprocess the image
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(target_side_len, target_side_len))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Rescale to [0,1]
    return img_array

# Path to the new image you want to test
new_img_path = 'Samples/test21.jpg'

# Preprocess the new image
preprocessed_img = preprocess_img('Samples/test21.jpg')

# Use the model to make predictions on the preprocessed image
predictions = loaded_model.predict(preprocessed_img)

# Print the predictions
if predictions < 0.50:
    print("Tampered Image")
if predictions >= 0.50:
    print("Real Image")
print(f"Decision Percentage: {predictions}")
