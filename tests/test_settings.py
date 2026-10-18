from dataclasses import asdict, replace
import unittest

from neo_sigma.settings import DEFAULTS, FULL_LABELS, LABELS, PREFIX, Settings, restore


class SettingsTests(unittest.TestCase):
    def test_basic_roundtrip_with_png_string_values(self):
        settings = Settings(enabled=True, start=.93, end=.03)
        self.assertEqual(restore({k: str(v) for k, v in settings.metadata().items()}), settings)
        self.assertTrue(settings.metadata()['Sigma Rescale'])

    def test_advanced_roundtrip_includes_all_controls(self):
        settings = Settings(True, .93, .03, True, .7, 1.1, True, 'Main + Hires', True, .6, .01, .8, .9, True)
        self.assertEqual(restore({k: str(v) for k, v in settings.metadata().items()}), settings)
        self.assertFalse(settings.metadata()['Sigma Rescale'])

    def test_plain_png_resets_every_setting(self):
        self.assertEqual(restore({'Steps': '30', 'CFG scale': '4'}), Settings())

    def test_simple_png_import(self):
        self.assertEqual(restore({'Sigma Rescale': 'True', 'Sigma Rescale start': '.93',
                                  'Sigma Rescale end': '.03'}), Settings(True, .93, .03))

    def test_full_png_import(self):
        settings = Settings(True, .93, .03, True, .5, 1.2, True, 'Main + Hires', True, .8, .02, .6, .9, False)
        params = {label: str(getattr(settings, key)) for key, label in FULL_LABELS.items()}
        self.assertEqual(restore(params), settings)

    def test_new_metadata_takes_precedence_including_disabled(self):
        self.assertFalse(restore({PREFIX: 'False', 'Sigma Rescale': 'True', 'Sigma Rescale Full': 'True'}).enabled)

    def test_basic_ignores_inactive_advanced_values(self):
        settings = Settings(True, .93, .03, False, -100, 100, True, 'Hires only', True)
        self.assertEqual(settings.for_pass(False).strength, 1)
        self.assertEqual(settings.for_pass(False).curve, 1)
        self.assertFalse(settings.for_pass(False).final_zero)
        self.assertIsNone(settings.for_pass(True))

    def test_pass_selection_and_separate_hires(self):
        settings = Settings(True, .93, .03, True, .7, 1.1, True, 'Main + Hires', True, .8, .02)
        self.assertEqual(settings.for_pass(False).start, .93)
        self.assertEqual(settings.for_pass(True).start, .8)
        self.assertEqual(replace(settings, separate_hires=False).for_pass(True), settings.for_pass(False))
        self.assertIsNone(replace(settings, passes='Hires only').for_pass(False))
        self.assertIsNone(replace(settings, passes='Main only').for_pass(True))

    def test_disabled_has_no_metadata_or_transform(self):
        settings = Settings(start=float('nan'))
        self.assertEqual(settings.metadata(), {})
        self.assertIsNone(settings.for_pass(False))
        self.assertIsNone(settings.for_pass(True))

    def test_malformed_enabled_metadata_rejected(self):
        for changes in ({LABELS['start']: 'oops'}, {LABELS['end']: 'NaN'},
                        {LABELS['advanced']: True, LABELS['curve']: 0},
                        {LABELS['advanced']: True, LABELS['passes']: 'bogus'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                restore({PREFIX: True, **changes})
