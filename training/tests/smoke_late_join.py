"""Real CPU late join: freeze a trained donor, inspect transfer, then average.

The child-only hooks create test barriers and snapshots without changing production
code. Files signal events; polling is bounded and is not used to order training.
"""
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import time

import torch


def wait_for(path, processes, timeout=120):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if any(p.poll() is not None for p in processes):
            raise RuntimeError(f"Process exited before {path.name}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path.name}")
        time.sleep(0.1)


def child():
    import hivemind
    import glmtrain.worker as worker

    root = Path(os.environ['LATE_JOIN_ROOT'])
    role = os.environ['LATE_JOIN_ROLE']

    def snapshot(optimizer, name):
        opt = optimizer.state_averager.optimizer
        torch.save({'parameters': [p.detach().cpu().clone() for group in opt.param_groups
                                   for p in group['params']],
                    'optimizer': optimizer.state_dict()}, root / (name + '.tmp'))
        os.replace(root / (name + '.tmp'), root / name)

    if role == 'donor':
        original = worker.save_checkpoint
        def save(*args):
            original(*args)
            if not (root / 'donor.pt').exists():
                snapshot(args[2], 'donor.pt')
                wait_for(root / 'release', [], 120)
        worker.save_checkpoint = save
    else:
        original = hivemind.Optimizer.load_state_from_peers
        def load(self, **kwargs):
            result = original(self, **kwargs)
            snapshot(self, 'joiner.pt')
            return result
        hivemind.Optimizer.load_state_from_peers = load
        original_batch = worker.TokenBatches.batch
        def batch(self, *args, **kwargs):
            if not (root / 'joiner-batch').exists():
                (root / 'joiner-batch').touch()
                wait_for(root / 'release', [], 120)
            return original_batch(self, *args, **kwargs)
        worker.TokenBatches.batch = batch
    runpy.run_module('glmtrain', run_name='__main__')


def assert_state_close(a, b, atol=0.002):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0.01, atol=atol)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for key in a:
            assert_state_close(a[key], b[key], atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_state_close(left, right, atol)
    else:
        assert a == b, (a, b)


def main():
    from glmtrain.data import prepare

    processes, handles = [], []
    with tempfile.TemporaryDirectory(prefix='glm-late-join-') as temp:
        root = Path(temp)
        (root / 'train.txt').write_text('A gator studies language by the lake.\n' * 400)
        (root / 'val.txt').write_text('A bird rests beside the river.\n' * 100)
        config = json.loads((Path(__file__).parents[1] / 'configs/smoke.json').read_text())
        config['training'].update(target_batch_tokens=16384, target_tokens=65536)
        (root / 'config.json').write_text(json.dumps(config))
        prepare(root / 'train.txt', root / 'val.txt', root / 'config.json', root / 'bundle')

        def launch(name, args, role=None):
            log = open(root / (name + '.log'), 'w')
            handles.append(log)
            env = {**os.environ, 'TOKENIZERS_PARALLELISM': 'false'}
            if role:
                env.update(LATE_JOIN_ROOT=str(root), LATE_JOIN_ROLE=role)
                command = [sys.executable, str(Path(__file__).resolve()), '--child', *args]
            else:
                command = [sys.executable, '-m', 'glmtrain', *args]
            p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
            processes.append(p)
            return p

        try:
            bootstrap = launch('bootstrap', ['bootstrap', '--bundle', str(root / 'bundle'),
                '--port', '0', '--identity', str(root / 'key'), '--peers-file', str(root / 'peers.json')])
            wait_for(root / 'peers.json', [bootstrap], 40)
            peer = json.loads((root / 'peers.json').read_text())[0]
            common = ['train', '--bundle', str(root / 'bundle'), '--peer', peer,
                '--device', 'cpu', '--microbatch', '1', '--threads', '1',
                '--matchmaking-time', '1', '--averaging-timeout', '30',
                '--join-timeout', '90', '--checkpoint-every', '1']
            donor = launch('donor', common + ['--name', 'donor', '--shard', '0',
                '--output', str(root / 'donor'), '--start-new'], 'donor')
            wait_for(root / 'donor.pt', [bootstrap, donor])
            joiner = launch('joiner', common + ['--name', 'joiner', '--shard', '1',
                '--output', str(root / 'joiner'), '--min-peers', '2'], 'joiner')
            wait_for(root / 'joiner.pt', [bootstrap, donor, joiner])
            a = torch.load(root / 'donor.pt', weights_only=True)
            b = torch.load(root / 'joiner.pt', weights_only=True)
            assert a['optimizer']['state']['local_epoch'] >= 1
            assert a['optimizer']['state']  # includes Adam moments, not just parameters
            assert any(isinstance(k, int) for k in a['optimizer']['state'])
            moments = [state["exp_avg"] for key, state in a["optimizer"]["state"].items()
                       if isinstance(key, int)]
            assert moments and any(t.abs().max() > 1e-5 for t in moments)
            assert_state_close(a["parameters"], b["parameters"])
            assert_state_close(a["optimizer"], b["optimizer"], atol=1e-7)
            wait_for(root / 'joiner-batch', [bootstrap, donor, joiner])
            (root / 'release').touch()
            for p in (donor, joiner):
                assert p.wait(timeout=150) == 0
            logs = (root / 'donor.log').read_text() + (root / 'joiner.log').read_text()
            assert 'Averaged gradients with 2 peers' in logs
            ready = [json.loads(line) for line in (root / 'joiner.log').read_text().splitlines()
                     if line.startswith('{') and '"event": "ready"' in line]
            assert ready and ready[0]['epoch'] >= 1 and ready[0]['shard'] == 1
            checkpoint = torch.load(root / 'joiner/latest.pt', weights_only=True)
            assert checkpoint['shard'] == 1 and checkpoint['local_tokens'] > 0
            assert checkpoint['optimizer']['state']['local_epoch'] > b['optimizer']['state']['local_epoch']
            print('PASS: trained peer model, Adam state and epoch transferred; late joiner subsequently averaged')
        except BaseException:
            for name in ('bootstrap', 'donor', 'joiner'):
                path = root / (name + '.log')
                if path.exists():
                    print(f'--- {name} ---\n' + path.read_text()[-16000:])
            raise
        finally:
            for p in reversed(processes):
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGINT)
            for p in processes:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            for handle in handles:
                handle.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        del sys.argv[1]
        child()
    else:
        main()
