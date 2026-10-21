import subprocess
import sys

import pytest


def command(body):
    return [sys.executable, "-u", "-c", body]


def test_worker_reuses_process_and_recovers_request_error():
    from asl_recognizer.translation.runtime_client import PersistentWorker

    body = """import json,sys,os
print(json.dumps({'ready':True,'protocol':1}),flush=True)
for line in sys.stdin:
 item=json.loads(line)
 reply={'id':item['id'],'ok':not item.get('fail'),
        'result':{'pid':os.getpid()},'error':'ValueError'}
 print(json.dumps(reply),flush=True)
"""
    with PersistentWorker(command(body), timeout=5) as worker:
        first = worker.request({})
        assert worker.request({}) == first
        with pytest.raises(ValueError, match="ValueError"):
            worker.request({"fail": True})
        assert worker.request({}) == first
    assert worker.process.poll() is not None


def test_worker_timeout_kills_process():
    from asl_recognizer.translation.runtime_client import PersistentWorker

    body = (
        "import json,time; print(json.dumps({'ready':True,'protocol':1}),flush=True);time.sleep(30)"
    )
    with (
        PersistentWorker(command(body), timeout=2) as worker,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        worker.request({}, timeout=0.05)
    assert worker.process.poll() is not None


@pytest.mark.parametrize("reply", ["print('noise',flush=True)", "print('{}',flush=True)", "pass"])
def test_worker_rejects_invalid_handshake(reply):
    from asl_recognizer.translation.runtime_client import PersistentWorker

    with pytest.raises(RuntimeError):
        PersistentWorker(command(reply), timeout=2)


def test_worker_rejects_mismatched_reply_id():
    from asl_recognizer.translation.runtime_client import PersistentWorker

    body = """import json,sys
print(json.dumps({'ready':True,'protocol':1}),flush=True)
sys.stdin.readline()
print(json.dumps({'id':'wrong','ok':True,'result':{}}),flush=True)
"""
    with (
        PersistentWorker(command(body), timeout=2) as worker,
        pytest.raises(RuntimeError, match="protocol"),
    ):
        worker.request({})
    assert worker.process.poll() is not None
