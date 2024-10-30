import keras
import tensorflow as tf
from keras_cv.layers import TransformerEncoder
from keras_cv.layers import PatchingAndEmbedding
from tensorflow.keras import layers,utils


@keras.saving.register_keras_serializable()
def parse_weights(weights, include_top, model_type):
    if not weights:
        return weights
    if weights.startswith("gs://"):
        weights = weights.replace("gs://", "https://storage.googleapis.com/")
        return utils.get_file(
            origin=weights,
            cache_subdir="models",
        )
    if tf.io.gfile.exists(weights):
        return weights
    if weights in ALIASES[model_type]:
        weights = ALIASES[model_type][weights]
    if weights in WEIGHTS_CONFIG[model_type]:
        if not include_top:
            weights = weights + "-notop"
        return utils.get_file(
            origin=f"{BASE_PATH}/{model_type}/{weights}.h5",
            cache_subdir="models",
            file_hash=WEIGHTS_CONFIG[model_type][weights],
        )

    raise ValueError(
        "The `weights` argument should be either `None`, a the path to the "
        "weights file to be loaded, or the name of pre-trained weights from "
        "https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/weights.py. "  # noqa: E501
        f"Invalid `weights` argument: {weights}"
    )


BASE_PATH = "https://storage.googleapis.com/keras-cv/models"

ALIASES = {
    "convmixer_512_16": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "cspdarknetl": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "cspdarknettiny": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "darknet53": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "deeplabv3": {
        "voc": "voc/segmentation-v0",
    },
    "densenet121": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "densenet169": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "densenet201": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "resnet50": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "resnet50v2": {
        "imagenet": "imagenet/classification-v2",
        "imagenet/classification": "imagenet/classification-v2",
    },
    "vittiny16": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "vits16": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "vitb16": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "vitl16": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "vits32": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "vitb32": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
}

WEIGHTS_CONFIG = {
    "convmixer_512_16": {
        "imagenet/classification-v0": "861f3080dc383f7936d3df89691aadea05eee6acaa4a0b60aa70dd657df915ee",  # noqa: E501
        "imagenet/classification-v0-notop": "aa08c7fa9ca6ec045c4783e1248198dbe1bc141e2ae788e712de471c0370822c",  # noqa: E501
    },
    "cspdarknetl": {
        "imagenet/classification-v0": "8bdc3359222f0d26f77aa42c4e97d67a05a1431fe6c448ceeab9a9c5a34ff804",  # noqa: E501
        "imagenet/classification-v0-notop": "9303aabfadffbff8447171fce1e941f96d230d8f3cef30d3f05a9c85097f8f1e",  # noqa: E501
    },
    "cspdarknettiny": {
        "imagenet/classification-v0": "c17fe6d7b597f2eb25e42fbd97ec58fb1dad753ba18920cc27820953b7947704",  # noqa: E501
        "imagenet/classification-v0-notop": "0007ae82c95be4d4aef06368a7c38e006381324d77e5df029b04890e18a8ad19",  # noqa: E501
    },
    "darknet53": {
        "imagenet/classification-v0": "7bc5589f7f7f7ee3878e61ab9323a71682bfb617eb57f530ca8757c742f00c77",  # noqa: E501
        "imagenet/classification-v0-notop": "8dcce43163e4b4a63e74330ba1902e520211db72d895b0b090b6bfe103e7a8a5",  # noqa: E501
    },
    "deeplabv3": {
        "voc/segmentation-v0": "732042e8b6c9ddba3d51c861f26dc41865187e9f85a0e5d43dfef75a405cca18",  # noqa: E501
    },
    "densenet121": {
        "imagenet/classification-v0": "13de3d077ad9d9816b9a0acc78215201d9b6e216c7ed8e71d69cc914f8f0775b",  # noqa: E501
        "imagenet/classification-v0-notop": "709afe0321d9f2b2562e562ff9d0dc44cca10ed09e0e2cfba08d783ff4dab6bf",  # noqa: E501
    },
    "densenet169": {
        "imagenet/classification-v0": "4cd2a661d0cb2378574073b23129ee4d06ea53c895c62a8863c44ee039e236a1",  # noqa: E501
        "imagenet/classification-v0-notop": "a99d1bb2cbe1a59a1cdd1f435fb265453a97c2a7b723d26f4ebee96e5fb49d62",  # noqa: E501
    },
    "densenet201": {
        "imagenet/classification-v0": "3b6032e744e5e5babf7457abceaaba11fcd449fe2d07016ae5076ac3c3c6cf0c",  # noqa: E501
        "imagenet/classification-v0-notop": "c1189a934f12c1a676a9cf52238e5994401af925e2adfc0365bad8133c052060",  # noqa: E501
    },
    "resnet50": {
        "imagenet/classification-v0": "1525dc1ce580239839ba6848c0f1b674dc89cb9ed73c4ed49eba355b35eac3ce",  # noqa: E501
        "imagenet/classification-v0-notop": "dc5f6d8f929c78d0fc192afecc67b11ac2166e9d8b9ef945742368ae254c07af",  # noqa: E501
    },
    "resnet50v2": {
        "imagenet/classification-v0": "11bde945b54d1dca65101be2648048abca8a96a51a42820d87403486389790db",  # noqa: E501
        "imagenet/classification-v0-notop": "5b4aca4932c433d84f6aef58135472a4312ed2fa565d53fedcd6b0c24b54ab4a",  # noqa: E501
        "imagenet/classification-v1": "a32e5d9998e061527f6f947f36d8e794ad54dad71edcd8921cda7804912f3ee7",  # noqa: E501
        "imagenet/classification-v1-notop": "ac46b82c11070ab2f69673c41fbe5039c9eb686cca4f34cd1d79412fd136f1ae",  # noqa: E501
        "imagenet/classification-v2": "5ee5a8ac650aaa59342bc48ffe770e6797a5550bcc35961e1d06685292c15921",  # noqa: E501
        "imagenet/classification-v2-notop": "e711c83d6db7034871f6d345a476c8184eab99dbf3ffcec0c1d8445684890ad9",  # noqa: E501
    },
    "vittiny16": {
        "imagenet/classification-v0": "c8227fde16ec8c2e7ab886169b11b4f0ca9af2696df6d16767db20acc9f6e0dd",  # noqa: E501
        "imagenet/classification-v0-notop": "aa4d727e3c6bd30b20f49d3fa294fb4bbef97365c7dcb5cee9c527e4e83c8f5b",  # noqa: E501
    },
    "vits16": {
        "imagenet/classification-v0": "4a66a1a70a879ff33a3ca6ca30633b9eadafea84b421c92174557eee83e088b5",  # noqa: E501
        "imagenet/classification-v0-notop": "8d0111eda6692096676a5453abfec5d04c79e2de184b04627b295f10b1949745",  # noqa: E501
    },
    "vitb16": {
        "imagenet/classification-v0": "6ab4e08c773e08de42023d963a97e905ccba710e2c05ef60c0971978d4a8c41b",  # noqa: E501
        "imagenet/classification-v0-notop": "4a1bdd32889298471cb4f30882632e5744fd519bf1a1525b1fa312fe4ea775ed",  # noqa: E501
    },
    "vitl16": {
        "imagenet/classification-v0": "5a98000f848f2e813ea896b2528983d8d956f8c4b76ceed0b656219d5b34f7fb",  # noqa: E501
        "imagenet/classification-v0-notop": "40d237c44f14d20337266fce6192c00c2f9b890a463fd7f4cb17e8e35b3f5448",  # noqa: E501
    },
    "vits32": {
        "imagenet/classification-v0": "f5836e3aff2bab202eaee01d98337a08258159d3b718e0421834e98b3665e10a",  # noqa: E501
        "imagenet/classification-v0-notop": "f3907845eff780a4d29c1c56e0ae053411f02fff6fdce1147c4c3bb2124698cd",  # noqa: E501
    },
    "vitb32": {
        "imagenet/classification-v0": "73025caa78459dc8f9b1de7b58f1d64e24a823f170d17e25fcc8eb6179bea179",  # noqa: E501
        "imagenet/classification-v0-notop": "f07b80c03336d731a2a3a02af5cac1e9fc9aa62659cd29e2e7e5c7474150cc71",  # noqa: E501
    },
}
@keras.saving.register_keras_serializable()
def parse_model_inputs(input_shape, input_tensor, **kwargs):
    if input_tensor is None:
        return tf.keras.layers.Input(shape=input_shape, **kwargs)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            return tf.keras.layers.Input(
                tensor=input_tensor, shape=input_shape, **kwargs
            )
        else:
            return input_tensor
MODEL_CONFIGS = {
    "ViTTiny16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 192,
        "mlp_dim": 768,
        "num_heads": 3,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTS16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTB16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTL16": {
        "patch_size": 16,
        "transformer_layer_num": 24,
        "project_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
    "ViTH16": {
        "patch_size": 16,
        "transformer_layer_num": 32,
        "project_dim": 1280,
        "mlp_dim": 5120,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
    "ViTTiny32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 192,
        "mlp_dim": 768,
        "num_heads": 3,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTS32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTB32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTL32": {
        "patch_size": 32,
        "transformer_layer_num": 24,
        "project_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
    "ViTH32": {
        "patch_size": 32,
        "transformer_layer_num": 32,
        "project_dim": 1280,
        "mlp_dim": 5120,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
}
@keras.saving.register_keras_serializable()
class ViT(keras.Model):
    def __init__(
        self,
        include_rescaling,
        include_top,
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        num_classes=None,
        patch_size=None,
        transformer_layer_num=None,
        num_heads=None,
        mlp_dropout=None,
        attention_dropout=None,
        activation=None,
        project_dim=None,
        mlp_dim=None,
        classifier_activation="softmax",
        **kwargs,
    ):
        if weights and not tf.io.gfile.exists(weights):
            raise ValueError(
                "The `weights` argument should be either `None` or the path "
                "to the weights file to be loaded. Weights file not found at "
                "location: {weights}"
            )

        if include_top and not num_classes:
            raise ValueError(
                "If `include_top` is True, you should specify `num_classes`. "
                f"Received: num_classes={num_classes}"
            )

        if include_top and pooling:
            raise ValueError(
                f"`pooling` must be `None` when `include_top=True`."
                f"Received pooling={pooling} and include_top={include_top}. "
            )

        inputs = parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

        # The previous layer rescales [0..255] to [0..1] if applicable
        # This one rescales [0..1] to [-1..1] since ViTs expect [-1..1]
        x = layers.Rescaling(scale=1.0 / 0.5, offset=-1.0, name="rescaling_2")(
            x
        )

        encoded_patches = PatchingAndEmbedding(project_dim, patch_size)(x)
        encoded_patches = layers.Dropout(mlp_dropout)(encoded_patches)

        for _ in range(transformer_layer_num):
            encoded_patches = TransformerEncoder(
                project_dim=project_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                mlp_dropout=mlp_dropout,
                attention_dropout=attention_dropout,
                activation=activation,
            )(encoded_patches)

        output = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        if include_top:
            output = output[:, 0]
            output = layers.Dense(
                num_classes, activation=classifier_activation
            )(output)

        elif pooling == "token_pooling":
            output = output[:, 0]
        elif pooling == "avg":
            output = layers.GlobalAveragePooling1D()(output)

        # Create model.
        super().__init__(inputs=inputs, outputs=output, **kwargs)

        if weights is not None:
            self.load_weights(weights)

        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.transformer_layer_num = transformer_layer_num
        self.num_heads = num_heads
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "include_rescaling": self.include_rescaling,
            "include_top": self.include_top,
            "name": self.name,
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "transformer_layer_num": self.transformer_layer_num,
            "num_heads": self.num_heads,
            "mlp_dropout": self.mlp_dropout,
            "attention_dropout": self.attention_dropout,
            "activation": self.activation,
            "project_dim": self.project_dim,
            "mlp_dim": self.mlp_dim,
            "classifier_activation": self.classifier_activation,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def ViTTiny16(
    *,
    include_rescaling,
    include_top,
    name="ViTTiny16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTTiny16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vittiny16"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTTiny16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTTiny16"][
            "transformer_layer_num"
        ],
        project_dim=MODEL_CONFIGS["ViTTiny16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTTiny16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTTiny16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTTiny16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTTiny16"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTS16(
    *,
    include_rescaling,
    include_top,
    name="ViTS16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTS16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vits16"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTS16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTB32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTS16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTS16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTS16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTS16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTS16"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTB16(
    *,
    include_rescaling,
    include_top,
    name="ViTB16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTB16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vitb16"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTB16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTB16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTB16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTB16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTB16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTB16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTB16"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTL16(
    *,
    include_rescaling,
    include_top,
    name="ViTL16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTL16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vitl16"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTL16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTL16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTL16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTL16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTL16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTL16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTL16"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTH16(
    *,
    include_rescaling,
    include_top,
    name="ViTH16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTH16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTH16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTH16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTH16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTH16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTH16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTH16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTH16"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTTiny32(
    *,
    include_rescaling,
    include_top,
    name="ViTTiny32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTTiny32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTTiny32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTTiny32"][
            "transformer_layer_num"
        ],
        project_dim=MODEL_CONFIGS["ViTTiny32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTTiny32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTTiny32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTTiny32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTTiny32"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTS32(
    *,
    include_rescaling,
    include_top,
    name="ViTS32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTS32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vits32"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTS32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTS32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTS32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTS32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTS32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTS32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTS32"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTB32(
    *,
    include_rescaling,
    include_top,
    name="ViTB32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTB32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vitb32"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTB32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTB32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTB32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTB32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTB32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTB32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTB32"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTL32(
    *,
    include_rescaling,
    include_top,
    name="ViTL32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTL32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTL32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTL32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTL32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTL32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTL32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTL32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTL32"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTH32(
    *,
    include_rescaling,
    include_top,
    name="ViTH32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTH32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTH32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTH32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTH32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTH32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTH32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTH32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTH32"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )
