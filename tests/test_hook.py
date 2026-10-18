from types import SimpleNamespace
import unittest
import torch

from neo_sigma.forge import OWNER_ATTR, ScheduleHook, prepare
from neo_sigma.settings import PREFIX, Settings
from neo_sigma.transform import Range, transform


class HookTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.stock = torch.tensor([1., .7, .3, 0.])
        calls, stock = self.calls, self.stock
        def original(sampler, processing, steps):
            calls.append(steps)
            return stock
        self.original = original
        self.cls = type('Sampler', (), {'get_sigmas': original})
        self.sampler = self.cls()
        self.p = SimpleNamespace(extra_generation_params={'Other extension': 'preserve'}, is_hr_pass=False)
        self.hook = ScheduleHook(self.cls)
        self.addCleanup(self.hook.close)

    def test_disabled_returns_original_object_and_calls_once(self):
        prepare(self.p, Settings())
        self.assertIs(self.sampler.get_sigmas(self.p, 3), self.stock)
        self.assertEqual(self.calls, [3])

    def test_basic_remap_with_metadata(self):
        prepare(self.p, Settings(True, .93, .03))
        actual = self.sampler.get_sigmas(self.p, 3)
        self.assertTrue(torch.equal(actual, transform(self.stock, Range(.93, .03))))
        params = self.p.extra_generation_params
        self.assertTrue(params[PREFIX + ' main applied'])
        self.assertEqual(len(params[PREFIX + ' main hash']), 12)
        self.assertEqual(params['Other extension'], 'preserve')
        self.assertTrue(params['Sigma Rescale'])
        self.assertEqual(self.calls, [3])

    def test_basic_bypasses_hires_exactly(self):
        prepare(self.p, Settings(True))
        self.p.is_hr_pass = True
        self.assertIs(self.sampler.get_sigmas(self.p, 3), self.stock)
        self.assertFalse(self.p.extra_generation_params[PREFIX + ' Hires applied'])

    def test_advanced_hires_has_own_hash(self):
        prepare(self.p, Settings(True, advanced=True, passes='Main + Hires', separate_hires=True, hires_start=.5))
        self.sampler.get_sigmas(self.p, 3)
        self.p.is_hr_pass = True
        actual = self.sampler.get_sigmas(self.p, 3)
        self.assertTrue(torch.equal(actual, transform(self.stock, Range(.5, 0))))
        params = self.p.extra_generation_params
        self.assertNotEqual(params[PREFIX + ' main hash'], params[PREFIX + ' Hires hash'])
        self.assertFalse(params['Sigma Rescale'])

    def test_zero_strength_is_reported_as_not_applied(self):
        prepare(self.p, Settings(True, advanced=True, strength=0))
        self.assertIs(self.sampler.get_sigmas(self.p, 3), self.stock)
        self.assertFalse(self.p.extra_generation_params[PREFIX + ' main applied'])

    def test_hires_only_does_not_enable_simple_reader(self):
        prepare(self.p, Settings(True, advanced=True, passes='Hires only'))
        self.assertIs(self.sampler.get_sigmas(self.p, 3), self.stock)
        self.assertFalse(self.p.extra_generation_params['Sigma Rescale'])

    def test_batch_diagnostics_do_not_carry_previous_hires_into_next_image(self):
        prepare(self.p, Settings(True, advanced=True, passes='Main + Hires'))
        self.sampler.get_sigmas(self.p, 3)
        self.p.is_hr_pass = True
        self.sampler.get_sigmas(self.p, 3)
        self.assertTrue(self.p.extra_generation_params[PREFIX + ' Hires applied'])
        self.p.is_hr_pass = False
        self.sampler.get_sigmas(self.p, 3)
        self.assertFalse(self.p.extra_generation_params[PREFIX + ' Hires applied'])
        self.assertNotIn(PREFIX + ' Hires hash', self.p.extra_generation_params)

    def test_disabling_clears_only_our_previous_metadata(self):
        prepare(self.p, Settings(True))
        self.sampler.get_sigmas(self.p, 3)
        prepare(self.p, Settings())
        self.assertEqual(self.p.extra_generation_params, {'Other extension': 'preserve'})

    def test_does_not_clear_legacy_metadata_we_did_not_emit(self):
        self.p.extra_generation_params['Sigma Rescale'] = True
        prepare(self.p, Settings())
        self.assertTrue(self.p.extra_generation_params['Sigma Rescale'])

    def test_conflicts_fail_before_inner_schedule_runs(self):
        prepare(self.p, Settings(True))
        for key, value in (('_sigma_rescale_standalone_enabled', True), ('_sigma_rescale_full_options', {'enabled': True})):
            setattr(self.p, key, value)
            with self.assertRaisesRegex(ValueError, 'disable the other'):
                self.sampler.get_sigmas(self.p, 3)
            delattr(self.p, key)
        self.assertEqual(self.calls, [])

    def test_invalid_process_settings_also_fail_at_sampling_boundary(self):
        with self.assertRaises(ValueError):
            prepare(self.p, Settings(True, start=float('nan')))
        with self.assertRaises(ValueError):
            self.sampler.get_sigmas(self.p, 3)
        self.assertEqual(self.calls, [])

    def test_close_restores_owned_method_and_is_idempotent(self):
        self.hook.close()
        self.hook.close()
        self.assertIs(self.cls.get_sigmas, self.original)
        self.assertFalse(hasattr(self.cls, OWNER_ATTR))

    def test_unload_preserves_later_wrapper_but_disables_ours(self):
        prepare(self.p, Settings(True))
        captured = self.cls.get_sigmas
        def later(sampler, processing, steps):
            return captured(sampler, processing, steps) + 1
        self.cls.get_sigmas = later
        self.hook.close()
        self.assertIs(self.cls.get_sigmas, later)
        self.assertTrue(torch.equal(self.sampler.get_sigmas(self.p, 3), self.stock + 1))

    def test_reload_does_not_double_apply_under_later_wrapper(self):
        prepare(self.p, Settings(True))
        captured = self.cls.get_sigmas
        def later(sampler, processing, steps):
            return captured(sampler, processing, steps)
        self.cls.get_sigmas = later
        replacement = ScheduleHook(self.cls)
        self.addCleanup(replacement.close)
        self.hook.close()  # stale unload callback must not remove the new owner
        self.assertIs(getattr(self.cls, OWNER_ATTR), replacement)
        self.assertTrue(torch.equal(self.sampler.get_sigmas(self.p, 3), transform(self.stock, Range())))
        self.assertEqual(self.calls, [3])
