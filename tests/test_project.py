import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

import cnn
import mlp


def create_digit_image(background, ink):
    image = Image.new("L", (160, 120), background)
    drawing = ImageDraw.Draw(image)
    drawing.line([(105, 15), (60, 70), (125, 70)], fill=ink, width=12)
    drawing.line([(105, 15), (105, 105)], fill=ink, width=12)
    return image


class ModelTests(unittest.TestCase):
    def check_model(self, model):
        images = torch.zeros(4, 1, 28, 28)
        logits = model(images)

        self.assertEqual(logits.shape, (4, 10))
        logits.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_mlp_forward_and_backward(self):
        self.check_model(mlp.MLP())

    def test_cnn_forward_and_backward(self):
        self.check_model(cnn.CNN())


class PreprocessingTests(unittest.TestCase):
    def test_cnn_uses_augmentation_only_for_training(self):
        train_transform = cnn.create_transform(train=True)
        test_transform = cnn.create_transform(train=False)

        self.assertTrue(
            any(
                isinstance(transform, transforms.RandomAffine)
                for transform in train_transform.transforms
            )
        )
        self.assertIsInstance(test_transform, transforms.ToTensor)

    def test_mlp_preprocessing_returns_expected_shape(self):
        image = create_digit_image(background=255, ink=0)
        image_batch, was_inverted = mlp.preprocess_image(image)

        self.assertEqual(image_batch.shape, (1, 1, 28, 28))
        self.assertTrue(was_inverted)
        self.assertGreaterEqual(image_batch.min().item(), 0.0)
        self.assertLessEqual(image_batch.max().item(), 1.0)

    def check_cnn_preprocessing(self, background, ink, expected_inversion):
        image = create_digit_image(background, ink)
        image_batch, was_inverted = cnn.preprocess_image(image)
        image_tensor = image_batch[0, 0]

        self.assertEqual(image_batch.shape, (1, 1, 28, 28))
        self.assertEqual(was_inverted, expected_inversion)
        border = torch.cat(
            (
                image_tensor[0, :],
                image_tensor[-1, :],
                image_tensor[:, 0],
                image_tensor[:, -1],
            )
        )
        self.assertEqual(border.max().item(), 0.0)

        rows, columns = torch.meshgrid(
            torch.arange(28),
            torch.arange(28),
            indexing="ij",
        )
        total_ink = image_tensor.sum()
        center_x = (columns * image_tensor).sum() / total_ink
        center_y = (rows * image_tensor).sum() / total_ink
        self.assertAlmostEqual(center_x.item(), 14.0, delta=1.0)
        self.assertAlmostEqual(center_y.item(), 14.0, delta=1.0)

    def test_cnn_preprocessing_handles_gray_background(self):
        self.check_cnn_preprocessing(110, 235, expected_inversion=False)

    def test_cnn_preprocessing_handles_white_background(self):
        self.check_cnn_preprocessing(255, 0, expected_inversion=True)

    def test_cnn_preprocessing_rejects_blank_image(self):
        with self.assertRaisesRegex(ValueError, "No handwritten digit"):
            cnn.preprocess_image(Image.new("L", (100, 100), 128))


class PredictionTests(unittest.TestCase):
    def check_prediction(self, module, model, image):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.pth"
            torch.save(model.state_dict(), model_path)

            with patch.object(module, "MODEL_PATH", model_path):
                prediction, confidence, image_batch, _ = module.predict(image)

        self.assertIn(prediction, range(10))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertEqual(image_batch.shape, (1, 1, 28, 28))

    def test_mlp_prediction_loads_state_dict(self):
        self.check_prediction(
            mlp,
            mlp.MLP(),
            create_digit_image(background=255, ink=0),
        )

    def test_cnn_prediction_loads_state_dict(self):
        self.check_prediction(
            cnn,
            cnn.CNN(),
            create_digit_image(background=110, ink=235),
        )


if __name__ == "__main__":
    unittest.main()
