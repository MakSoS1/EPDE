"""TF1 graph API for DISCOVER's fixed-library, attention=False benchmark mode.

Used only in the isolated DISCOVER interpreter. The RNN, policy-gradient
training and sparse regression are upstream DISCOVER code. Legacy Keras
provides the original TF1 cells/layers that Keras 3 removed.
"""
import sys
import types
import importlib.abc
import importlib.machinery


class _LegacySource(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Compile in memory: the checksum-verified upstream source stays intact.
        source = self.get_data(self.path).decode('utf-8')
        source = source.replace('.tostring()', '.tobytes()')
        return self.source_to_code(source, self.path)


class _DiscoverImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'dso' and not fullname.startswith('dso.'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = _LegacySource(fullname, spec.origin)
        return spec


def activate(root):
    sys.meta_path.insert(0, _DiscoverImports())
    import collections
    import collections.abc
    collections.Mapping = collections.abc.Mapping
    import numpy as np
    import tensorflow as tensorflow
    import tensorflow.compat.v1 as tf
    from tf_keras.src.layers.rnn.legacy_cells import LSTMCell, GRUCell, MultiRNNCell
    from tf_keras.src.layers.rnn.legacy_cells import LayerRNNCell
    from tf_keras.src.legacy_tf_layers import core

    tf.disable_v2_behavior()
    # DISCOVER imports tensorflow at module scope; keep TensorFlow internals
    # intact and supply its v1 names on the real module.
    for name in dir(tf):
        if not hasattr(tensorflow, name):
            setattr(tensorflow, name, getattr(tf, name))
    tensorflow.nn = tf.nn
    tensorflow.train = tf.train
    tensorflow.summary = tf.summary
    tensorflow.losses = tf.losses
    tensorflow.metrics = tf.metrics
    tensorflow.layers = types.SimpleNamespace(dense=core.dense)
    tensorflow.nn.rnn_cell = types.SimpleNamespace(LSTMCell=LSTMCell, GRUCell=GRUCell)
    tensorflow.contrib = types.SimpleNamespace(
        rnn=types.SimpleNamespace(LayerRNNCell=LayerRNNCell, MultiRNNCell=MultiRNNCell),
        layers=types.SimpleNamespace(variance_scaling_initializer=lambda factor, mode, uniform, seed:
            tf.variance_scaling_initializer(scale=factor, mode=mode.lower(),
                                           distribution='uniform' if uniform else 'truncated_normal', seed=seed)))
    # PDEComp explicitly disables attention. Do not import its unused TF1-only
    # attention implementation (which imports the removed contrib package).
    attention = types.ModuleType('dso.attention')

    def unavailable(*args, **kwargs):
        raise ValueError('DISCOVER compatibility mode requires attention=False')

    attention.AttentionCellWrapper = unavailable
    sys.modules['dso.attention'] = attention
    for name, value in (('int', int), ('float', float), ('bool', np.bool_)):
        if name not in np.__dict__:
            setattr(np, name, value)
