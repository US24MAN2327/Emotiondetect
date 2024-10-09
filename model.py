
import tensorflow as tf### models



from tensorflow.keras.models import load_model




IMG = 255


# Now you can load the model without needing the custom object scope

import os
model_path = os.path.join(os.getcwd(), 'seq.keras')
resnet34 = load_model(model_path)


# Print the model summary
print(resnet34.summary())
