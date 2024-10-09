
import tensorflow as tf### models



from tensorflow.keras.models import load_model




IMG = 255


# Now you can load the model without needing the custom object scope
resnet34 = load_model('seq.keras')

# Print the model summary
print(resnet34.summary())
