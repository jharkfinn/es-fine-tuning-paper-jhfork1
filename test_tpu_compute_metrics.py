#!/usr/bin/env python3
"""
Test script to explore TPU compute metrics available from vLLM actor.
"""

import os
import ray

@ray.remote
class TpuMetricsExplorer:
    def __init__(self):
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        os.environ.setdefault("VLLM_DEVICE", "tpu")

    def explore_metrics(self):
        """Explore all available TPU metrics from JAX/XLA."""
        import pprint
        try:
            import jax
            import jax.numpy as jnp

            results = {}

            # Device info
            devices = jax.devices()
            results['num_devices'] = len(devices)
            results['device_info'] = []

            for i, dev in enumerate(devices):
                dev_info = {
                    'index': i,
                    'platform': dev.platform,
                    'device_kind': dev.device_kind,
                    'attributes': [attr for attr in dir(dev) if not attr.startswith('_')]
                }

                # Memory stats
                if hasattr(dev, 'memory_stats'):
                    dev_info['memory_stats'] = dev.memory_stats()

                # Client info
                if hasattr(dev, 'client'):
                    client = dev.client
                    dev_info['client_attrs'] = [attr for attr in dir(client)
                                                 if not attr.startswith('_') and
                                                 ('stat' in attr.lower() or 'metric' in attr.lower())]

                results['device_info'].append(dev_info)

            # XLA client
            if hasattr(jax.lib, 'xla_client'):
                xla_client = jax.lib.xla_client
                results['xla_client_attrs'] = [
                    attr for attr in dir(xla_client)
                    if not attr.startswith('_') and
                    ('stat' in attr.lower() or 'metric' in attr.lower() or 'util' in attr.lower())
                ]

            # Backend info
            try:
                from jax.extend import backend as jax_backend
                backend = jax_backend.get_backend()
                results['backend'] = {
                    'platform': backend.platform,
                    'attrs': [attr for attr in dir(backend)
                             if not attr.startswith('_') and
                             ('stat' in attr.lower() or 'metric' in attr.lower())]
                }
            except Exception as backend_err:
                results['backend'] = {'error': str(backend_err)}

            # Profiler APIs
            if hasattr(jax, 'profiler'):
                results['profiler_attrs'] = [attr for attr in dir(jax.profiler)
                                             if not attr.startswith('_')]

            # Run a test computation and check metrics
            x = jnp.ones((1000, 1000))
            y = jnp.dot(x, x)
            jax.block_until_ready(y)

            if hasattr(devices[0], 'memory_stats'):
                results['memory_after_compute'] = devices[0].memory_stats()

            return results

        except Exception as e:
            return {'error': str(e), 'traceback': __import__('traceback').format_exc()}

if __name__ == '__main__':
    # Initialize Ray
    ray.init(address="local", ignore_reinit_error=True)

    # Create actor with TPU runtime
    explorer = TpuMetricsExplorer.options(
        runtime_env={
            "env_vars": {
                "TPU_VISIBLE_CHIPS": "0",
                "PJRT_DEVICE": "TPU",
                "VLLM_DEVICE": "tpu",
            },
            "pip": ["jax[tpu]", "numpy"],
        }
    ).remote()

    # Run exploration
    print("Exploring TPU metrics from actor...")
    results = ray.get(explorer.explore_metrics.remote())

    # Print results
    import pprint
    print("\n" + "=" * 80)
    print("TPU Metrics Exploration Results:")
    print("=" * 80)
    pprint.pprint(results, width=100)

    ray.shutdown()
