import os
import sys

import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.learning_configurator import LearningConfigurator


def _model():
    # encoder_weights=None: no download, and the freezing logic does not care
    # what the values are.
    return smp.Unet(encoder_name="mit_b5", encoder_weights=None, in_channels=3, classes=1)


def _trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_head_only_trains_nothing_but_the_head():
    model = LearningConfigurator().prepare_model_for_head_only(_model())
    assert _trainable(model.encoder) == 0
    assert _trainable(model.decoder) == 0
    assert _trainable(model.segmentation_head) > 0


def test_transfer_learning_trains_decoder_and_head_only():
    model = LearningConfigurator().prepare_model_for_transfer_learning(_model())
    assert _trainable(model.encoder) == 0
    assert _trainable(model.decoder) == sum(p.numel() for p in model.decoder.parameters())
    assert _trainable(model.segmentation_head) > 0


def test_first_conv_helper_warns_instead_of_silently_doing_nothing(capsys):
    LearningConfigurator().prepare_model_for_transfer_learning(_model())
    assert "patch_embed1" in capsys.readouterr().out


def test_block4_is_a_valid_unfreeze_key():
    model = LearningConfigurator().prepare_model_for_fine_tuning(_model(), ["block4"])
    assert _trainable(model.encoder.block4) == sum(
        p.numel() for p in model.encoder.block4.parameters()
    )
    assert _trainable(model.encoder.block1) == 0
