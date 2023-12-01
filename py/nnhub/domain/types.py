from enum import StrEnum


class ModelType(StrEnum):
    IMAGE_CLASSIFICATION = 'image_classification_model'
    LARGE_LANGUAGE = 'large_language_model'
    SPEECH_TO_TEXT = 'speech_to_text_model'