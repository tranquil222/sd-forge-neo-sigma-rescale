import importlib.util
from pathlib import Path
import sys
import unittest
import torch

from neo_sigma.comfy import SigmaRescaleNeo
from neo_sigma.transform import Range, transform


class ComfyTests(unittest.TestCase):
    def test_registration_by_package_loader(self):
        path = Path(__file__).resolve().parents[1] / '__init__.py'
        spec = importlib.util.spec_from_file_location('neo_comfy_test', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(module)
        self.assertIn('SigmaRescaleNeo', module.NODE_CLASS_MAPPINGS)

    def test_node_basic_and_advanced_match_core(self):
        source = torch.tensor([1., .7, .2, 0.])
        node = SigmaRescaleNeo()
        self.assertTrue(torch.equal(node.apply(source, start=.93, end=.03)[0], transform(source, Range(.93, .03))))
        self.assertTrue(torch.equal(node.apply(source, advanced=True, strength=.6, curve=1.1, final_zero=True)[0],
                                    transform(source, Range(.9, 0, .6, 1.1, True))))
        self.assertIs(node.apply(source, enabled=False)[0], source)

    def test_basic_ignores_advanced_settings(self):
        source = torch.tensor([1., 0.])
        self.assertTrue(torch.equal(SigmaRescaleNeo().apply(source, curve=-1, strength=-1)[0], transform(source, Range())))
