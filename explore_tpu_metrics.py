#!/usr/bin/env python3
"""
Explore available TPU metrics from JAX/XLA APIs.
"""

import pprint

try:
    import jax
    import jax.numpy as jnp

    print("=" * 80)
    print("JAX Devices:")
    print("=" * 80)
    devices = jax.devices()
    for i, dev in enumerate(devices):
        print(f"\nDevice {i}: {dev}")
        print(f"  Platform: {dev.platform}")
        print(f"  Device kind: {dev.device_kind}")

        # List all attributes
        print(f"\n  Available attributes:")
        for attr in dir(dev):
            if not attr.startswith('_'):
                print(f"    - {attr}")

        # Try memory_stats
        if hasattr(dev, 'memory_stats'):
            print(f"\n  Memory stats:")
            mem_stats = dev.memory_stats()
            pprint.pprint(mem_stats, indent=4)

        # Try client
        if hasattr(dev, 'client'):
            print(f"\n  Client: {dev.client}")
            client = dev.client
            for attr in dir(client):
                if not attr.startswith('_') and 'stat' in attr.lower():
                    print(f"    Client stat attribute: {attr}")

    print("\n" + "=" * 80)
    print("JAX Profiler APIs:")
    print("=" * 80)
    if hasattr(jax, 'profiler'):
        print("Available in jax.profiler:")
        for attr in dir(jax.profiler):
            if not attr.startswith('_'):
                print(f"  - {attr}")

    print("\n" + "=" * 80)
    print("XLA Client APIs:")
    print("=" * 80)
    if hasattr(jax.lib, 'xla_client'):
        xla_client = jax.lib.xla_client
        print("Available in xla_client:")
        for attr in dir(xla_client):
            if not attr.startswith('_') and ('stat' in attr.lower() or 'metric' in attr.lower() or 'util' in attr.lower()):
                print(f"  - {attr}")

        # Try to get device stats
        if hasattr(xla_client, 'Device'):
            print("\nXLA Device attributes:")
            for attr in dir(xla_client.Device):
                if not attr.startswith('_'):
                    print(f"  - {attr}")

    print("\n" + "=" * 80)
    print("Compilation and Execution Counters:")
    print("=" * 80)

    # Try JAX compilation cache
    if hasattr(jax, '_src'):
        try:
            from jax._src import compilation_cache
            print("Compilation cache available")
        except:
            pass

    # Try getting backend
    backend = jax.lib.xla_bridge.get_backend()
    print(f"\nBackend: {backend}")
    print(f"Platform: {backend.platform}")

    for attr in dir(backend):
        if not attr.startswith('_') and ('stat' in attr.lower() or 'metric' in attr.lower()):
            print(f"  Backend attribute: {attr}")

    print("\n" + "=" * 80)
    print("Testing simple computation to check runtime metrics:")
    print("=" * 80)

    # Run a simple computation
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    jax.block_until_ready(y)

    # Check if metrics changed
    if hasattr(devices[0], 'memory_stats'):
        print("\nMemory stats after computation:")
        pprint.pprint(devices[0].memory_stats(), indent=4)

    # Try to access profiler trace
    print("\n" + "=" * 80)
    print("Checking for profiler trace APIs:")
    print("=" * 80)

    if hasattr(jax.profiler, 'trace'):
        print("jax.profiler.trace available - can use for detailed profiling")
        print("Usage: with jax.profiler.trace('/tmp/prof'): ...")

except ImportError as e:
    print(f"JAX not available: {e}")
    print("\nThis is expected in the driver environment.")
    print("Run this script inside a vLLM actor to access JAX/TPU metrics.")
