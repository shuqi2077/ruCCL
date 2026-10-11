"""Run real selected-backend collectives on an explicit list of GPU devices."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing
import time
from . import Communicator, rendezvous


def rank_check(rank, devices, session, backend, algorithm, path):
    device_class, index = devices[rank]
    world = len(devices)
    started = time.perf_counter()
    with Communicator(rank, world, session, backend=backend, device=index,
                      device_class=device_class, algorithm=algorithm, library_path=path) as group:
        def check(operation, values, expected, root=0):
            with group.upload(values) as buffer:
                result = group.collective(buffer, operation, root=root)
                try:
                    actual = result.download().tolist()
                    assert actual == expected, (operation, rank, actual[:8], expected[:8])
                finally:
                    if result is not buffer: result.close()
        offset = 100*world*(world-1)//2
        check(0, [i+rank*100 for i in range(257)], [i*world+offset for i in range(257)])
        check(1, [i+rank*100 for i in range(7)], [i+(world-1)*100 for i in range(7)])
        check(2, [i+rank*100 for i in range(7)], list(range(7)))
        check(3, [i+rank*100 for i in range(3)], [i+(world-1)*100 for i in range(3)], world-1)
        check(4, [i+rank*100 for i in range(3)], [i+r*100 for r in range(world) for i in range(3)])
        check(5, [i+rank*100 for i in range(3*world)], [i*world+offset for i in range(rank*3, rank*3+3)])
        with group.upload([i+rank*100 for i in range(13)]) as buffer:
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(group.collective, buffer, 0).result()
            assert buffer.download().tolist() == [i*world+offset for i in range(13)]
        return dict(rank=rank, backend=backend, algorithm=algorithm,
            device_class=device_class, device_index=index, checks=7,
            elapsed_seconds=time.perf_counter()-started, communication=group.stats())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, choices=('rocm', 'metal', 'vulkan'))
    parser.add_argument('--devices', required=True, help='rank-ordered class:index entries, e.g. discrete:0,discrete:1')
    parser.add_argument('--algorithm', default='auto', choices=('auto', 'ring'))
    parser.add_argument('--library')
    args = parser.parse_args()
    devices = []
    for entry in args.devices.split(','):
        kind, index = entry.split(':')
        devices.append((kind, int(index)))
    session = rendezvous(len(devices), library_path=args.library)
    print('QWEN_EVENT '+json.dumps(dict(stage='collectives',completed=0,total=7*len(devices))),flush=True)
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(rank_check, rank, devices, session,
                   args.backend, args.algorithm, args.library) for rank in range(len(devices))]
        results = []
        for future in futures:
            results.append(future.result())
            print('QWEN_EVENT '+json.dumps(dict(stage='collectives',completed=7*len(results),total=7*len(devices))),flush=True)
    print('RUCCL_RESULT '+json.dumps(dict(status='passed',ranks=results)),flush=True)


if __name__ == '__main__':
    main()
