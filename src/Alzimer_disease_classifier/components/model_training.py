from Alzimer_disease_classifier.entity.config_entity import TrainingConfig, DataIngestionConfig
import tensorflow as tf
from pathlib import Path
from Alzimer_disease_classifier.utils.common import copy_model
from typing import Optional
from Alzimer_disease_classifier.logger import logging

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model: Optional[tf.keras.Model] = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        """Load base model with error handling and verification"""
        try:
            self.model = tf.keras.models.load_model(
                self.config.updated_base_model_path,
                compile=False  # Start with fresh compilation
            )
            # Verify model structure
            if not hasattr(self.model, 'layers'):
                raise ValueError("Invalid model format - loaded object is not a Keras model")
            
            logging.info(f"Successfully loaded base model from {self.config.updated_base_model_path}")
            logging.debug(f"Model summary:\n{self.model.summary()}")
            
        except Exception as e:
            logging.error(f"Failed to load base model: {str(e)}")
            raise

    def _configure_data_generators(self):
        """Configure data augmentation and preprocessing"""
        base_datagen_kwargs = {
            'rescale': 1.0/255,
            'validation_split': 0.30,
            'fill_mode': 'nearest'
        }

        # Validation generator (no augmentation)
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**base_datagen_kwargs)
        
        # Training generator with conditional augmentation
        if self.config.params_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **base_datagen_kwargs
            )
        else:
            train_datagen = valid_datagen

        return train_datagen, valid_datagen

    def train_valid_generator(self):
        """Create data generators with enhanced validation"""
        try:
            train_datagen, valid_datagen = self._configure_data_generators()

            common_flow_args = {
                'directory': self.config.training_data,
                'target_size': self.config.params_image_size[:-1],
                'batch_size': self.config.params_batch_size,
                'interpolation': 'bicubic',
                'class_mode': 'categorical'
            }

            self.train_generator = train_datagen.flow_from_directory(
                subset='training',
                shuffle=True,
                seed=42,
                **common_flow_args
            )

            self.valid_generator = valid_datagen.flow_from_directory(
                subset='validation',
                shuffle=False,
                seed=42,
                **common_flow_args
            )

            # Verify class consistency
            if self.train_generator.class_indices != self.valid_generator.class_indices:
                raise ValueError("Class indices mismatch between train and validation generators")

            logging.info(f"Class labels: {self.train_generator.class_indices}")

        except Exception as e:
            logging.error(f"Data generator setup failed: {str(e)}")
            raise

    def _compile_model(self):
        """Compile model with dynamic configuration"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save model in modern Keras format"""
        model.save(path.with_suffix('.keras'), save_format='keras',model=model)

    def train(self):
        """Enhanced training process with callbacks and monitoring"""
        try:
            # Calculate steps with ceiling
            self.steps_per_epoch = -(-self.train_generator.samples // self.train_generator.batch_size)  # Ceiling division
            self.validation_steps = -(-self.valid_generator.samples // self.valid_generator.batch_size)

            # Add callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=3,
                    monitor='val_loss',
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.config.trained_model_path,
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1
                )
            ]

            # Compile model
            self._compile_model()

            # Training process
            history = self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_data=self.valid_generator,
                validation_steps=self.validation_steps,
                callbacks=callbacks,
                verbose=2
            )
    
            self.save_model(path=self.config.trained_model_path, model=self.model)
            logging.info(f"Model training completed successfully")
            return history

        except Exception as e:
            logging.error(f"Training failed: {str(e)}")

   