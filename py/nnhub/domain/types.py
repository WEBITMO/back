from enum import Enum


class ModelType(str, Enum):
    IMAGE_CLASSIFICATION = 'image_classification_model'
    IMAGE_SEGMENTATION = 'image_segmentation_model'
    LARGE_LANGUAGE = 'large_language_model'
    SPEECH_TO_TEXT = 'speech_to_text_model'
