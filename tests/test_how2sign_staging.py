from __future__ import annotations

import hashlib
import zlib
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest

from asl_recognizer.translation.how2sign_staging import stage_recordings
from asl_recognizer.translation.manifest import build_manifest, write_manifest
from asl_recognizer.translation.sol_common import read_json, write_json


def setup(tmp_path: Path):  # type: ignore[no-untyped-def]
    mirror, recovery = tmp_path / "mirror", tmp_path / "recovered"
    mirror.mkdir()
    recovery.mkdir()
    archives = []
    names = [("train", "one-rgb_front"), ("dev", "two-rgb_front"), ("test", "test-rgb_front")]
    for split, name in names:
        relative = f"{'val' if split == 'dev' else split}_raw_videos/{split}.zip"
        archive = mirror / relative
        archive.parent.mkdir()
        with ZipFile(archive, "w", compression=ZIP_STORED) as bundle:
            bundle.writestr(name + ".mp4", b"video:" + name.encode())
        write_json(archive.with_suffix('.verified.json'), {
            'bytes': archive.stat().st_size,
            'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'recordings': [name + '.mp4']})
        archives.append({'path': relative, 'recordings': [name + '.mp4']})
    catalogue = tmp_path / 'catalogue.json'
    write_json(catalogue, {'archives': archives})
    status = tmp_path / 'recovery.json'
    write_json(status, {'status': 'completed', 'completed': []})
    manifest = tmp_path / 'manifest.csv'
    write_manifest(manifest, build_manifest([{
        'clip_id': split + ':' + name, 'video_id': name, 'split': split,
        'caption': 'hello', 'start_ms': 0, 'end_ms': 1000,
    } for split, name in names], source='how2sign'))
    return dict(manifest_path=manifest, catalogue_path=catalogue, mirror_root=mirror,
                recovery_root=recovery, recovery_status=status, output=tmp_path / 'out',
                remaining_quota=100000)


def test_stages_train_dev_only_and_repairs_output(tmp_path: Path) -> None:
    options = setup(tmp_path)
    first = stage_recordings(**options)
    output = options['output'] / 'train/one-rgb_front.mp4'
    assert output.read_bytes() == b'video:one-rgb_front'
    assert len(first['recordings']) == 2
    assert not (options['output'] / 'test').exists()
    before = output.stat().st_mtime_ns
    second = stage_recordings(**options)
    assert output.stat().st_mtime_ns == before
    assert second['reused'] == 2
    output.write_bytes(b'corrupt')
    assert stage_recordings(**options)['reused'] == 1
    assert output.read_bytes() == b'video:one-rgb_front'
    receipt = read_json(output.with_suffix('.verified.json'))
    assert receipt['source']['kind'] == 'mirror'
    assert receipt['seal']['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'crosssplit', 'traversal'])
def test_rejects_ambiguous_missing_or_unsafe_catalogue(tmp_path: Path, change: str) -> None:
    options = setup(tmp_path)
    catalogue = read_json(options['catalogue_path'])
    if change == 'missing':
        catalogue['archives'][0]['recordings'] = []
    elif change == 'duplicate':
        catalogue['archives'].append(catalogue['archives'][0])
    elif change == 'crosssplit':
        catalogue['archives'][1]['recordings'].append('one-rgb_front.mp4')
    else:
        catalogue['archives'][0]['path'] = '../train.zip'
    write_json(options['catalogue_path'], catalogue)
    with pytest.raises(ValueError):
        stage_recordings(**options)


@pytest.mark.parametrize('change', ['sha', 'size', 'crc', 'member_traversal'])
def test_archive_integrity_and_member_paths(tmp_path: Path, change: str) -> None:
    options = setup(tmp_path)
    archive = options['mirror_root'] / 'train_raw_videos/train.zip'
    receipt_path = archive.with_suffix('.verified.json')
    receipt = read_json(receipt_path)
    if change == 'sha':
        receipt['sha256'] = '0' * 64
    elif change == 'size':
        receipt['bytes'] += 1
    elif change == 'crc':
        data = archive.read_bytes().replace(b'video:one', b'XXXXX:one')
        archive.write_bytes(data)
        receipt['sha256'] = hashlib.sha256(data).hexdigest()
    else:
        with ZipFile(archive, 'w') as bundle:
            bundle.writestr('../one-rgb_front.mp4', b'video')
        receipt['bytes'] = archive.stat().st_size
        receipt['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    write_json(receipt_path, receipt)
    with pytest.raises((ValueError, OSError)):
        stage_recordings(**options)


def test_capacity_rejection_before_staging(tmp_path: Path) -> None:
    options = setup(tmp_path)
    options['remaining_quota'] = 1
    with pytest.raises(ValueError, match='insufficient capacity'):
        stage_recordings(**options)
    assert not list(options['output'].rglob('*.mp4'))


@pytest.mark.parametrize('change', ['valid', 'receipt', 'sha', 'crc', 'size'])
def test_recovered_recording_requires_matching_receipt_and_integrity(
    tmp_path: Path, change: str,
) -> None:
    options = setup(tmp_path)
    catalogue = read_json(options['catalogue_path'])
    catalogue['archives'] = catalogue['archives'][1:]
    write_json(options['catalogue_path'], catalogue)
    source = options['recovery_root'] / 'train/one-rgb_front.mp4'
    source.parent.mkdir()
    data = b'recovered full recording'
    source.write_bytes(data)
    row = {'filename': source.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
           'crc32': zlib.crc32(data), 'split': 'train'}
    write_json(options['recovery_status'], {'status': 'completed', 'completed': [row]})
    receipt = dict(row)
    if change == 'receipt':
        receipt['sha256'] = '0' * 64
    elif change == 'sha':
        source.write_bytes(data.upper())
    elif change == 'crc':
        row['crc32'] += 1
        receipt['crc32'] += 1
        write_json(options['recovery_status'], {'status': 'completed', 'completed': [row]})
    elif change == 'size':
        source.write_bytes(data + b'extra')
    write_json(source.with_suffix('.verified.json'), receipt)
    if change == 'valid':
        result = stage_recordings(**options)
        assert len(result['recordings']) == 2
        assert (options['output'] / 'train' / source.name).read_bytes() == data
    else:
        with pytest.raises(ValueError):
            stage_recordings(**options)
