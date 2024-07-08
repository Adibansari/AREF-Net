import sys
from pathlib import Path

# Add the project's root directory to the sys.path
root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))
import sys
from pathlib import Path
from src.models.model import create_arefnet
from src.data.make_dataset import load_data
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import yaml

def train_model(params):
    input_shape = (32, 32, 3)
    num_classes = 10
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) =load_data()

    # Normalize the dataset
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # Create the AREF-Net model
    model = create_arefnet(input_shape, num_classes)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)
    
    # Train the model
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=params['train']['batch_size']),
        validation_data=(x_test, y_test),
        epochs=params['train']['epochs'],
        verbose=1
    )
    
    # Save the model
    
    model_path = Path('models')
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path / 'arefnet_model.pkl')
    joblib.dump(history.history, model_path / 'arefnet_model_history.pkl')

if __name__ == '__main__':
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    train_model(params)
