import unittest
import torch

from neo_sigma.transform import Range, transform


class TransformTests(unittest.TestCase):
    def test_linear_arithmetic_is_exact_and_input_is_unchanged(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for start, end in ((0.93, 0.03), (0.9, 0), (0.1, 0.95), (-0.5, -2), (1, 1)):
                with self.subTest(dtype=dtype, start=start, end=end):
                    source = torch.linspace(0, 1, 26, dtype=dtype).flip(0)
                    saved = source.clone()
                    a, b = source.new_tensor(start), source.new_tensor(end)
                    expected = ((source - source.min()) * (a - b)) / (source.max() - source.min()) + b
                    result = transform(source, Range(start, end))
                    self.assertTrue(torch.equal(result, expected))
                    self.assertTrue(torch.equal(source, saved))
                    self.assertEqual(result.dtype, source.dtype)
                    self.assertEqual(result.device, source.device)

    def test_power_curve_and_blend(self):
        source = torch.tensor([10., 5., 0.], dtype=torch.float64)
        self.assertTrue(torch.equal(transform(source, Range(1, 0, curve=2)), torch.tensor([1, .25, 0])))
        target = torch.tensor([1., .25, 0.], dtype=torch.float64)
        self.assertTrue(torch.equal(transform(source, Range(1, 0, .5, 2)), source + .5 * (target - source)))

    def test_strength_zero_is_exact_bypass_even_with_final_zero(self):
        source = torch.tensor([1., .5, .1])
        self.assertIs(transform(source, Range(strength=0, final_zero=True)), source)

    def test_empty_single_constant_are_exact_bypass(self):
        for source in (torch.tensor([]), torch.tensor([.5]), torch.ones(4)):
            self.assertIs(transform(source, Range(final_zero=True)), source)

    def test_final_zero_only_changes_last_value(self):
        source = torch.tensor([1., .7, .3, 0.])
        expected = transform(source, Range(.93, .03))
        actual = transform(source, Range(.93, .03, final_zero=True))
        self.assertTrue(torch.equal(actual[:-1], expected[:-1]))
        self.assertEqual(actual[-1], 0)
        self.assertGreater(expected[-1], 0)

    def test_noncontiguous_input(self):
        source = torch.linspace(1, 0, 14)[::2]
        self.assertFalse(source.is_contiguous())
        self.assertEqual(transform(source, Range()).shape, source.shape)

    def test_invalid_input_and_numeric_overflow_are_rejected(self):
        for source in ([1., 0.], torch.ones(2, 2), torch.tensor([1, 0]), torch.tensor([1., float('nan')]),
                       torch.tensor([float('inf'), 0.])):
            with self.subTest(source=source), self.assertRaises(ValueError):
                transform(source, Range())
        with self.assertRaises(ValueError):
            transform(torch.tensor([1., 0.], dtype=torch.float16), Range(1e30, 0))

    def test_invalid_settings_are_rejected(self):
        for settings in (Range(curve=0), Range(curve=11), Range(strength=-1), Range(strength=2),
                         Range(start=float('nan')), Range(end=float('inf'))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                transform(torch.tensor([1., 0.]), settings)
