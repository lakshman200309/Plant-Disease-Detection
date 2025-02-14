from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Load base model without top layers
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

# Fix the output shape issue
x = base_model.output
x = GlobalAveragePooling2D()(x)  # This reduces 25088 -> 2048
x = Dense(1024, activation="relu")(x)
output = Dense(5, activation="softmax")(x)  # Adjust for the number of classes

# Create and compile model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Save the corrected model
model.save("inceptionv3_plant_disease_fixed.h5")

print("✅ Model saved as inceptionv3_plant_disease_fixed.h5")
