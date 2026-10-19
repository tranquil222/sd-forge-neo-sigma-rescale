"""Late timing and interoperability; no model loading or image generation."""
import importlib.util
import math
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from neo_sigma.comfy import SigmaRescaleNeo
from neo_sigma.forge import ScheduleHook, prepare
from neo_sigma.settings import Settings, restore
from neo_sigma.transform import Range, transform


class TimingTests(unittest.TestCase):
    def test_prefix_including_boundary_is_bit_exact_for_odd_step_counts(self):
        for steps in (5, 15, 25, 40):
            source = torch.linspace(1, 0, steps + 1)
            for percent in (1, 50, 60, 85):
                count = math.ceil(steps * percent / 100)
                result = transform(source, Range(1, 0, curve=1.75, begin_percent=percent))
                self.assertTrue(torch.equal(source[:count + 1], result[:count + 1]))
                self.assertEqual(len(source), len(result))

    def test_ramp_and_counted_suffix(self):
        source = torch.tensor([1., .875, .75, .625, .5, .375, .25, .125, 0.], dtype=torch.float64)
        # Four sampled intervals, first two unchanged: boundary is index 6.
        result = transform(source, Range(1, 0, curve=2, begin_percent=50), sample_start=4)
        self.assertTrue(torch.equal(result[:7], source[:7]))
        self.assertEqual(result[7].item(), .125 + .5 * (.125**2 - .125))
        self.assertEqual(result[8].item(), 0)

    def test_identity_modes_preserve_nonzero_terminal_sigma(self):
        source = torch.tensor([1., .7, .3, .1])
        for settings in (Range(begin_percent=100, final_zero=True),
                         Range(begin_percent=99, final_zero=True),
                         Range(begin_percent=50, strength=0, final_zero=True)):
            self.assertIs(transform(source, settings), source)

    def test_bad_percentages_and_rising_schedules_fail(self):
        source = torch.tensor([1., .8, .4, .1, 0.])
        for percent in (-1, 101, float('nan')):
            with self.assertRaises(ValueError):
                transform(source, Range(begin_percent=percent))
        with self.assertRaisesRegex(ValueError, 'descending'):
            transform(source.flip(0), Range(begin_percent=50))
        with self.assertRaisesRegex(ValueError, 'increase sigma'):
            transform(source, Range(1, .5, begin_percent=50))

    def test_full_png_late_values_and_old_missing_values(self):
        png = {'Sigma Rescale Full': True, 'Sigma Rescale Full start': 1,
               'Sigma Rescale Full end': .02, 'Sigma Rescale Full curve': 1.75,
               'Sigma Rescale Full begin percent': 60,
               'Sigma Rescale Full Hires begin percent': 75}
        selected = restore(png)
        self.assertTrue(selected.advanced)
        self.assertEqual(selected.begin_percent, 60)
        self.assertEqual(selected.hires_begin_percent, 75)
        self.assertEqual(restore(selected.metadata()), selected)
        png.pop('Sigma Rescale Full begin percent')
        png.pop('Sigma Rescale Full Hires begin percent')
        self.assertEqual(restore(png).begin_percent, 0)
        self.assertEqual(restore(png).hires_begin_percent, 0)

    def test_comfy_optional_input_keeps_old_workflows_and_matches_late_core(self):
        source = torch.linspace(1, 0, 16)
        node = SigmaRescaleNeo()
        self.assertTrue(torch.equal(node.apply(source, advanced=True, curve=1.75, begin_percent=60)[0],
                                   transform(source, Range(curve=1.75, begin_percent=60))))
        self.assertTrue(torch.equal(node.apply(source, begin_percent=60)[0], node.apply(source)[0]))


class Img2imgTimingTests(unittest.TestCase):
    def test_request_local_context_counts_actual_suffix_and_restores_on_error(self):
        stock = torch.linspace(1, 0, 13)
        helper = ModuleType('modules.sd_samplers_common')
        helper.setup_img2img_steps = lambda p, steps=None: (12, 3 if steps is None else steps - 1)
        modules = ModuleType('modules')
        modules.sd_samplers_common = helper

        class Sampler:
            def get_sigmas(self, p, steps):
                if getattr(p, 'fail', False):
                    raise RuntimeError('test interruption')
                return stock

            def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning,
                              steps=None, image_conditioning=None):
                planned, encoded = helper.setup_img2img_steps(p, steps)
                return self.get_sigmas(p, planned)[planned - encoded - 1:]

        native = Sampler.sample_img2img
        sampler = Sampler()
        hook = ScheduleHook(Sampler)
        self.addCleanup(hook.close)
        p = SimpleNamespace(extra_generation_params={})
        selected = Settings(True, 1, 0, advanced=True, curve=2, begin_percent=50)
        prepare(p, selected)
        with patch.dict(sys.modules, {'modules': modules, 'modules.sd_samplers_common': helper}):
            for explicit, start in ((None, 8), (6, 6)):
                actual = sampler.sample_img2img(p, None, None, None, None, steps=explicit)
                expected = transform(stock, selected.for_pass(False), sample_start=start)[start:]
                self.assertTrue(torch.equal(actual, expected))
                self.assertIsNone(hook.img2img_context.get())
            p.fail = True
            with self.assertRaises(RuntimeError):
                sampler.sample_img2img(p, None, None, None, None)
            self.assertIsNone(hook.img2img_context.get())
            p.fail = False
            self.assertTrue(torch.equal(sampler.get_sigmas(p, 12), transform(stock, selected.for_pass(False))))
        hook.close()
        self.assertIs(Sampler.sample_img2img, native)

    def test_endpoint_only_sampler_rejected_without_affecting_legacy_mode(self):
        class Sampler:
            func = staticmethod(lambda model, x, sigma_min, sigma_max: None)
            def get_sigmas(self, p, steps):
                return torch.linspace(1, 0, steps + 1)
        hook = ScheduleHook(Sampler)
        self.addCleanup(hook.close)
        p = SimpleNamespace(extra_generation_params={})
        prepare(p, Settings(True, advanced=True, begin_percent=50))
        with self.assertRaisesRegex(ValueError, 'explicit sigma schedule'):
            Sampler().get_sigmas(p, 10)
        prepare(p, Settings(True))
        self.assertEqual(len(Sampler().get_sigmas(p, 10)), 11)


@unittest.skipUnless(os.environ.get('SIGMA_FULL_REFERENCE'), 'Optional local Full reference is not bundled')
class LegacyFullParityTests(unittest.TestCase):
    def test_exact_schedule_parity_with_existing_full_images(self):
        spec = importlib.util.spec_from_file_location('local_full_reference', os.environ['SIGMA_FULL_REFERENCE'])
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        compared = 0
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for count in (8, 15, 25, 40):
                source = torch.linspace(1, 0, count + 1, dtype=dtype)
                for begin in (0, 25, 50, 60, 85, 100):
                    for start, end, strength, curve, zero in ((.93, .03, 1, 1, False),
                            (1, .02, 1, 1.75, False), (.9, 0, .5, 3, True), (1, .04, .5, 3, False)):
                        for offset in (0, count // 2):
                            with self.subTest(dtype=dtype, count=count, begin=begin, curve=curve, offset=offset):
                                settings = Range(start, end, strength, curve, zero, begin)
                                try:
                                    expected = reference.rescale_sigmas(source, start, end, strength, curve,
                                        force_zero=zero, begin_percent=begin, sample_start=offset)
                                except ValueError:
                                    with self.assertRaises(ValueError):
                                        transform(source, settings, sample_start=offset)
                                else:
                                    self.assertTrue(torch.equal(expected, transform(source, settings, sample_start=offset)))
                                compared += 1
        self.assertEqual(compared, 768)
