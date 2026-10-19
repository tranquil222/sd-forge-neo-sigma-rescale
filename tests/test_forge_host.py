"""Optional CPU integration checks against a user's existing Forge checkout.

Set FORGE_ROOT to enable. Methods are read from that checkout, not bundled here;
we do not import the full application, load models, or start a GPU job.
"""
import ast
from contextlib import contextmanager
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import torch

from neo_sigma.forge import ScheduleHook, prepare
from neo_sigma.settings import DEFAULTS, FULL_LABELS, PREFIX, Settings, restore
from neo_sigma.transform import Range, transform

ROOT = Path(os.environ['FORGE_ROOT']) if os.environ.get('FORGE_ROOT') else None
PROJECT = Path(__file__).resolve().parents[1]


def definitions(relative_path, names, namespace, class_name=None):
    source = ROOT / relative_path
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    if class_name:
        tree = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    body = [item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name in names]
    if {item.name for item in body} != set(names):
        raise AssertionError(f'Host definitions changed: {names}')
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


@unittest.skipUnless(ROOT, 'Set FORGE_ROOT for CPU host integration checks')
class HostTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from scipy import stats
        self.opts = SimpleNamespace(beta_dist_alpha=.6, beta_dist_beta=.6,
            always_discard_next_to_last_sigma=False, sigma_min=0, sigma_max=0)
        names = {'simple_scheduler', 'beta_scheduler', 'to_d'}
        sched = definitions('modules/sd_schedulers.py', names,
                            dict(torch=torch, np=np, stats=stats, shared=SimpleNamespace(opts=self.opts)))
        self.schedulers = {name: SimpleNamespace(label=name, function=sched[fn], default_rho=-1,
                                                need_inner_model=True)
                           for name, fn in [('Simple', 'simple_scheduler'), ('Beta', 'beta_scheduler')]}
        method = definitions('modules/sd_samplers_kdiffusion.py', {'get_sigmas'}, dict(
            torch=torch, opts=self.opts, devices=SimpleNamespace(cpu='cpu'),
            sd_schedulers=SimpleNamespace(schedulers_map=self.schedulers)), 'KDiffusionSampler')['get_sigmas']
        self.cls = type('Sampler', (), {'get_sigmas': method})
        self.native = method
        self.sampler = self.cls()
        self.sampler.config = SimpleNamespace(options={})
        self.sampler.model_wrap = SimpleNamespace(sigmas=torch.linspace(.001, 1, 1000))
        self.p = SimpleNamespace(is_hr_pass=False, scheduler='Beta', hr_scheduler='Simple',
                                 sampler_noise_scheduler_override=None, extra_generation_params={})
        self.hook = ScheduleHook(self.cls)
        self.addCleanup(self.hook.close)
        self.euler = definitions('modules_forge/packages/k_diffusion/sampling.py', {'sample_euler'},
            dict(torch=torch, to_d=sched['to_d'], trange=lambda count, **kwargs: range(count)))['sample_euler']

    def test_native_schedulers_and_euler_sample_match_linear_reference(self):
        for scheduler in self.schedulers:
            for steps in (8, 12, 25, 40):
                for start, end in ((.9, 0), (.93, .03)):
                    with self.subTest(scheduler=scheduler, steps=steps, start=start, end=end):
                        self.p.scheduler = scheduler
                        native = self.native(self.sampler, self.p, steps)
                        prepare(self.p, Settings(True, start, end))
                        result = self.sampler.get_sigmas(self.p, steps)
                        a, b = native.new_tensor(start), native.new_tensor(end)
                        expected = ((native-native.min()) * (a-b)) / (native.max()-native.min()) + b
                        self.assertTrue(torch.equal(result, expected))
                        noise = torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(7))
                        model = lambda x, sigma, **kwargs: x * .25
                        actual_image = self.euler(model, noise * result[0], result, disable=True)
                        expected_image = self.euler(model, noise * expected[0], expected, disable=True)
                        self.assertTrue(torch.equal(actual_image, expected_image))
                        self.assertTrue(torch.isfinite(actual_image).all())

    def test_native_override_and_penultimate_removal_precede_transform(self):
        supplied = torch.tensor([1., .6, .4, .1, 0.])
        self.p.sampler_noise_scheduler_override = lambda steps: supplied
        self.opts.always_discard_next_to_last_sigma = True
        prepare(self.p, Settings(True, .93, .03))
        actual = self.sampler.get_sigmas(self.p, 3)
        self.assertTrue(torch.equal(actual, transform(supplied[[0, 1, 2, 4]], Range(.93, .03))))

    def test_native_hires_bypass(self):
        self.p.is_hr_pass = True
        prepare(self.p, Settings(True))
        self.assertTrue(torch.equal(self.sampler.get_sigmas(self.p, 8), self.native(self.sampler, self.p, 8)))

    def test_img2img_still_slices_after_constructing_schedule(self):
        source = ast.parse((ROOT / 'modules/sd_samplers_kdiffusion.py').read_text(encoding='utf-8-sig'))
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'KDiffusionSampler')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_img2img')
        assignments = [n for n in method.body if isinstance(n, ast.Assign)]
        sigma = next(n for n in assignments if any(isinstance(t, ast.Name) and t.id == 'sigmas' for t in n.targets))
        sliced = next(n for n in assignments if any(isinstance(t, ast.Name) and t.id == 'sigma_sched' for t in n.targets))
        self.assertLess(sigma.lineno, sliced.lineno)
        self.assertIn('self.get_sigmas', ast.unparse(sigma.value))
        self.assertIsInstance(sliced.value, ast.Subscript)

    def test_png_file_through_native_parser_and_real_gradio_paste_fields(self):
        import gradio as gr
        from PIL import Image, PngImagePlugin
        from neo_sigma.ui import build_controls
        source = (ROOT / 'modules/infotext_utils.py').read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        constants = {}
        for item in tree.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == 're_param_code':
                        constants['re_param'] = re.compile(ast.literal_eval(item.value))
        namespace = dict(gr=gr, Any=Any, json=json, os=os, type_of_gr_update=type(gr.skip()),
            shared=SimpleNamespace(opts=SimpleNamespace(infotext_skip_pasting=[], strip_whitespaces=True,
                                                         sd_model_checkpoint='test.safetensors')),
            errors=SimpleNamespace(report=lambda *a, **k: self.fail('Native paste callback failed')),
            re_imagesize=re.compile(r'^(\d+)x(\d+)$'),
            _extract_styles=lambda params, prompt, negative: (prompt, negative),
            _populate_defaults=lambda params: None, uses_emphasis=lambda text: False,
            main_entry=SimpleNamespace(module_list={}), **constants)
        definitions('modules/infotext_utils.py', {'parse_generation_parameters', '_parse_info', 'unquote'}, namespace)
        with gr.Blocks(analytics_enabled=False):
            controls, fields = build_controls()
        basic = Settings(True, .93, .03)
        advanced = Settings(True, .93, .03, True, .7, 1.1, True, 'Main + Hires', True, .8, .01, .6, .9, True,
                            begin_percent=60, hires_begin_percent=75)
        cases = [(basic, basic.metadata()), (advanced, advanced.metadata()), (Settings(), {})]
        cases.append((basic, {'Sigma Rescale': True, 'Sigma Rescale start': .93, 'Sigma Rescale end': .03}))
        cases.append((advanced, {label: getattr(advanced, key) for key, label in FULL_LABELS.items()}))
        for settings, metadata in cases:
            with self.subTest(settings=settings):
                values = dict(Steps=8, Sampler='Euler', Seed=7, **metadata)
                text = 'A blue ceramic cup.\n' + ', '.join(f'{k}: {v}' for k, v in values.items())
                png_info = PngImagePlugin.PngInfo()
                png_info.add_text('parameters', text)
                buffer = io.BytesIO()
                Image.new('RGB', (8, 8), '#4488aa').save(buffer, format='PNG', pnginfo=png_info)
                buffer.seek(0)
                with Image.open(buffer) as image:
                    parsed = namespace['parse_generation_parameters'](image.info['parameters'])
                pasted = [namespace['_parse_info'](component, target, parsed) for component, target in fields]
                self.assertEqual(Settings(**dict(zip(DEFAULTS, [v['value'] for v in pasted[:len(DEFAULTS)]]))), settings)
                self.assertEqual(pasted[-2]['visible'], settings.advanced)
                self.assertEqual(pasted[-1]['visible'], settings.advanced and settings.separate_hires)

    def test_actual_entrypoint_registration_and_reload(self):
        modules = ModuleType('modules')
        modules.scripts = SimpleNamespace(Script=type('BaseScript', (), {}), AlwaysVisible=object())
        unloaders = []
        modules.script_callbacks = SimpleNamespace(on_script_unloaded=unloaders.append)
        native_module = ModuleType('modules.sd_samplers_kdiffusion')
        native_module.KDiffusionSampler = self.cls
        path = PROJECT / 'scripts/neo_sigma_extension.py'
        with patch.dict(sys.modules, {'modules': modules, 'modules.sd_samplers_kdiffusion': native_module}):
            for index in range(2):
                spec = importlib.util.spec_from_file_location(f'neo_script_test_{index}', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.addCleanup(module.HOOK.close)
                script = module.Script()
                script.process(self.p, True, .93, .03)
                native = self.native(self.sampler, self.p, 8)
                self.assertTrue(torch.equal(self.sampler.get_sigmas(self.p, 8), transform(native, Range(.93, .03))))
                self.assertIs(script.show(False), modules.scripts.AlwaysVisible)
            unloaders[0]()
            self.assertTrue(module.HOOK.active)
