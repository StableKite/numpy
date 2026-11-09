"""Integrity, metadata-only installation and source fallback contracts."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from zipfile import ZipFile
import importlib.util
import io
import json
import shutil
import tempfile
import unittest

backend_path=Path(__file__).with_name('numpy_stablekite_build_backend.py')
if not backend_path.exists():
    backend_path=Path(__file__).resolve().parents[2]/'_stablekite_build_backend.py'
spec=importlib.util.spec_from_file_location('numpy_carrier',backend_path)
backend=importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'pyproject.toml').write_text('[project]\nname="numpy"\nversion="2.5.4+stablekite.1"\n[tool.stablekite]\nsource-build-requires=["meson-python>=0.18.0","Cython>=3.1.0"]\n')
        self.version='2.5.4+stablekite.1'
        self.commit='a'*40
        self.tag='cp315-cp315t-win_amd64'
        self.filename=f'numpy-{self.version}-{self.tag}.whl'
        self.dist_info=f'numpy-{self.version}.dist-info'
        self.native='numpy/_core/_multiarray_umath.cp315t-win_amd64.pyd'
        self.make_wheel()
        self.root_patch=mock.patch.object(backend,'ROOT',self.root)
        self.root_patch.start(); self.addCleanup(self.root_patch.stop)
        self.platform_patch=mock.patch.object(backend,'_platform_key',return_value='win_amd64')
        self.platform_patch.start(); self.addCleanup(self.platform_patch.stop)

    def make_wheel(self,metadata_version=None,runtime_version=None,commit=None,extra=None):
        path=self.root/self.filename
        with ZipFile(path,'w') as wheel:
            wheel.writestr(self.dist_info+'/METADATA',f'Metadata-Version: 2.4\nName: numpy\nVersion: {metadata_version or self.version}\nRequires-Python: >=3.12\n\n')
            wheel.writestr(self.dist_info+'/WHEEL',f'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {self.tag}\n\n')
            wheel.writestr(self.dist_info+'/RECORD','')
            wheel.writestr('numpy/version.py',f'version={runtime_version or self.version!r}\ngit_revision={commit or self.commit!r}\n')
            wheel.writestr(self.native,b'qualified native fixture')
            if extra: wheel.writestr(extra,b'forbidden metadata path')
        self.wheel=path
        self.artifact={'filename':self.filename,'url':f'https://github.com/StableKite/numpy/releases/download/test/{self.filename}',
                       'sha256':sha256(path.read_bytes()).hexdigest(),'size':path.stat().st_size,'tag':self.tag,
                       'native_hashes':{self.native:sha256(b'qualified native fixture').hexdigest()}}
        self.manifest={'schema_version':1,'version':self.version,'native_source_commit':self.commit,
                       'artifacts':{'win_amd64':self.artifact}}
        self.write_manifest()

    def write_manifest(self):
        (self.root/'stablekite-wheels.json').write_text(json.dumps(self.manifest))

    def test_verified_metadata_and_wheel_reuse_one_download(self):
        with mock.patch.object(backend.tempfile,'gettempdir',return_value=str(self.root)), \
             mock.patch.object(backend.urllib.request,'urlopen',return_value=io.BytesIO(self.wheel.read_bytes())) as download:
            name=backend.prepare_metadata_for_build_wheel(str(self.root/'metadata'))
            self.assertEqual(name,self.dist_info)
            built=backend.build_wheel(str(self.root/'output'))
        self.assertEqual(download.call_count,1)
        self.assertEqual(built,self.filename)
        self.assertEqual((self.root/'output'/built).read_bytes(),self.wheel.read_bytes())
        self.assertEqual((self.root/'metadata'/name/'METADATA').read_bytes(),ZipFile(self.wheel).read(name+'/METADATA'))
        self.assertEqual(backend.get_requires_for_build_wheel(),[])

    def test_corrupt_download_is_rejected_without_cache_artifact(self):
        with mock.patch.object(backend.tempfile,'gettempdir',return_value=str(self.root)), \
             mock.patch.object(backend.urllib.request,'urlopen',return_value=io.BytesIO(b'corrupted')):
            with self.assertRaisesRegex(RuntimeError,'SHA-256 or size mismatch'):
                backend.build_wheel(str(self.root/'output'))
        self.assertFalse(list((self.root/'stablekite-numpy-wheel-cache').glob('*.whl')))
        self.assertFalse(list((self.root/'stablekite-numpy-wheel-cache').glob('*.download')))

    def test_native_binary_hash_is_checked(self):
        self.artifact['native_hashes'][self.native]='0'*64
        with self.assertRaisesRegex(RuntimeError,'native binary hash mismatch'):
            backend._validate_wheel(self.wheel,self.manifest,self.artifact)

    def test_metadata_version_must_match_real_runtime_and_project(self):
        self.make_wheel(metadata_version='2.6.0')
        with self.assertRaisesRegex(RuntimeError,'metadata differs'):
            backend._validate_wheel(self.wheel,self.manifest,self.artifact)
        self.make_wheel(runtime_version='2.6.0')
        with self.assertRaisesRegex(RuntimeError,'runtime version/source commit differs'):
            backend._validate_wheel(self.wheel,self.manifest,self.artifact)

    def test_native_source_commit_must_match(self):
        self.make_wheel(commit='b'*40)
        with self.assertRaisesRegex(RuntimeError,'runtime version/source commit differs'):
            backend._validate_wheel(self.wheel,self.manifest,self.artifact)

    def test_metadata_cannot_escape_preparation_directory(self):
        self.make_wheel(extra=self.dist_info+'/../../escaped')
        with mock.patch.object(backend,'_wheel',return_value=(self.wheel,self.manifest,self.artifact)):
            with self.assertRaisesRegex(RuntimeError,'Invalid path'):
                backend.prepare_metadata_for_build_wheel(str(self.root/'metadata'))
        self.assertFalse((self.root/'escaped').exists())

    def test_source_metadata_uses_meson_once_and_reuses_exact_wheel(self):
        calls=[]
        def build(folder,settings):
            calls.append(settings)
            shutil.copyfile(self.wheel,Path(folder)/self.filename)
            return self.filename
        meson=SimpleNamespace(build_wheel=build)
        settings={'stablekite-source-build':'true','build-dir':'native-build'}
        with mock.patch.object(backend.importlib,'import_module',return_value=meson):
            metadata=backend.prepare_metadata_for_build_wheel(str(self.root/'metadata'),settings)
            built=backend.build_wheel(str(self.root/'output'),settings,str(self.root/'metadata'/metadata))
        self.assertEqual(calls,[{'build-dir':'native-build'}])
        self.assertEqual((self.root/'output'/built).read_bytes(),self.wheel.read_bytes())
        requirements=backend.get_requires_for_build_wheel(settings)
        self.assertIn('meson-python>=0.18.0',requirements)
        self.assertIn('Cython==3.3.1',requirements)

    def test_source_cache_integrity_failure_does_not_silently_rebuild(self):
        def build(folder,settings):
            shutil.copyfile(self.wheel,Path(folder)/self.filename)
            return self.filename
        settings={'stablekite-source-build':'true'}
        with mock.patch.object(backend.importlib,'import_module',return_value=SimpleNamespace(build_wheel=build)):
            metadata=backend.prepare_metadata_for_build_wheel(str(self.root/'metadata'),settings)
        (self.root/'metadata/_stablekite_source_wheel'/self.filename).write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'prepared source wheel hash mismatch'):
            backend.build_wheel(str(self.root/'output'),settings,str(self.root/'metadata'/metadata))

    def test_noncarrier_interpreters_use_original_source_requirements(self):
        with mock.patch.object(backend,'_platform_key',return_value=None):
            self.assertIn('meson-python>=0.18.0',backend.get_requires_for_build_wheel())
        with self.assertRaises(ValueError):
            backend.get_requires_for_build_wheel({'stablekite-source-build':'maybe'})


if __name__=='__main__': unittest.main()
