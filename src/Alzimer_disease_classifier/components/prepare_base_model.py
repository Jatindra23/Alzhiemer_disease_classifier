import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import sys
from pathlib import Path
from Alzimer_disease_classifier.exception import RenelException
from Alzimer_disease_classifier.logger import logging
from Alzimer_disease_classifier.entity.config_entity import PrepareBaseModelConfig
from typing import Optional



    
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model: Optional[tf.keras.Model] = None
        self.full_model: Optional[tf.keras.Model] = None

    def get_base_model(self):
        """Load pre-trained VGG16 with custom configuration"""
        self.model = tf.keras.applications.VGG16(
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            input_shape=self.config.params_image_size,
            pooling='max'  # Add global max pooling
        )
        self._freeze_initial_layers()

    def _freeze_initial_layers(self):
        """Freeze convolutional blocks for transfer learning"""
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer.trainable = False

    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model,
        classes: int,
        freeze_till: Optional[int],
        learning_rate: float
        
    ) -> tf.keras.Model:
        """Enhanced model builder with regularization and fine-tuning"""
        # Fine-tuning configuration
        if freeze_till is not None:
            for layer in model.layers[:freeze_till]:
                layer.trainable = False
            for layer in model.layers[freeze_till:]:
                layer.trainable = True

        # Custom head with regularization
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        predictions = tf.keras.layers.Dense(
            units=classes,
            activation='softmax' if classes > 1 else 'sigmoid'
        )(x)

        full_model = tf.keras.Model(inputs=model.input, outputs=predictions)

        # Optimizer with learning rate schedule
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=1000,
                decay_rate=0.96
            )
        )

        # Configure loss based on problem type
        loss = (tf.keras.losses.CategoricalCrossentropy() 
                if classes > 1 
                else tf.keras.losses.BinaryCrossentropy())

        full_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        full_model.summary()
        return full_model

    def update_base_model(self):
        """Create the full model with custom head"""
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_till=self.config.params_freeze_till,
            learning_rate=self.config.params_learning_rate,
            
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save model in modern Keras format"""
        model.save(path.with_suffix('.keras'), save_format='keras',model=model) 
