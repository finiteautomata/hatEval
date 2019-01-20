from .base_model import BaseModel
from keras.layers import (
    Dense, Concatenate
)
import keras


class MergeModel(BaseModel):
    def __init__(self, models, **kwargs):

        self.models = models
        # Build the graph
        inputs = []
        to_merge = []
        for model in models:
            inputs.append(model.layers[0].input)
            to_merge.append(model.layers[-2].output)

        if len(models) > 1:
            merge_layer = Concatenate()(to_merge)
        else:
            merge_layer = outputs[0]
        output = Dense(1, activation='sigmoid')(merge_layer)

        super().__init__(inputs=[inputs], outputs=[output], **kwargs)

    def preprocess_fit(self, X):
        for model in self.models:
            model.preprocess_fit(X)

    def preprocess_transform(self, X):
        return [model.preprocess_transform(X) for model in self.models]
