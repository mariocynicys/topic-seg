from tensorflow_models import nlp
from tensorflow import keras


@keras.saving.register_keras_serializable("encoder")
class EncoderStack(keras.layers.Layer):
    """A layer that consists of `num_layers` transformer encoders connected to each others."""
    def __init__(self, num_layers: int, **kwargs):
        super().__init__()
        self._transformer_layers = [
            nlp.layers.TransformerEncoderBlock(
                inner_activation="relu",
                **kwargs)
            for _ in range(num_layers)
        ]

    def call(self, inputs):
        for transformer in self._transformer_layers:
            inputs = transformer(inputs)
        # TODO: Consider normalizing the last output.
        return inputs

    def get_config(self):
        """This method is crucial for this layer to be serializable/storable."""
        transformer_layer_config = self._transformer_layers[0].get_config()
        # Pop the `inner_activation` since its already hard-coded to `relu` above.
        transformer_layer_config.pop("inner_activation")
        return {
            "num_layers": len(self._transformer_layers),
            **transformer_layer_config
        }
