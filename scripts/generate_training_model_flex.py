#!/usr/bin/env python3
"""
Generates test/assets/training_model_flex.tflite

A minimal linear regression model (y = wx + b) with Google's standard
on-device training signatures: train, infer, save, restore. Uses
SELECT_TF_OPS for tf.raw_ops.SaveV2/RestoreV2 checkpoint persistence
(V2 format: creates .index + .data-00000-of-00001 files).

Requirements:
    pip install tensorflow>=2.13

Usage:
    python scripts/generate_training_model_flex.py

Model properties:
    - train:   inputs x [1,1] float32, y [1,1] float32  -> loss [1] float32
    - infer:   input  x [1,1] float32                   -> output [1,1] float32
    - save:    input  checkpoint_path [1] string         -> status [1] int32
    - restore: input  checkpoint_path [1] string         -> status [1] int32

Initial weights: w=0, b=0
"""

import os
import sys
import numpy as np
import tensorflow as tf


class LinearModelWithCheckpoint(tf.Module):
    def __init__(self):
        self.w = tf.Variable([[0.0]], dtype=tf.float32, name='weight')
        self.b = tf.Variable([0.0], dtype=tf.float32, name='bias')

    @tf.function(input_signature=[
        tf.TensorSpec([1, 1], tf.float32, name='x'),
        tf.TensorSpec([1, 1], tf.float32, name='y'),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            pred = tf.matmul(x, self.w) + self.b
            loss = tf.reduce_mean(tf.square(pred - y))
        grads = tape.gradient(loss, [self.w, self.b])
        self.w.assign_sub(0.01 * grads[0])
        self.b.assign_sub(0.01 * grads[1])
        return {'loss': loss}

    @tf.function(input_signature=[
        tf.TensorSpec([1, 1], tf.float32, name='x'),
    ])
    def infer(self, x):
        return {'output': tf.matmul(x, self.w) + self.b}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1], dtype=tf.string, name='checkpoint_path'),
    ])
    def save(self, checkpoint_path):
        path = checkpoint_path[0]
        save_op = tf.raw_ops.SaveV2(
            prefix=path,
            tensor_names=tf.constant(['weight', 'bias']),
            shape_and_slices=tf.constant(['', '']),
            tensors=[self.w.read_value(), self.b.read_value()],
        )
        # Wrap the return in a control dependency on save_op so that the
        # TFLite MLIR dead-code-elimination pass cannot remove SaveV2
        # (which has no output tensors and would otherwise be eliminated).
        with tf.control_dependencies([save_op]):
            return {'status': tf.identity(tf.constant(0, dtype=tf.int32))}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1], dtype=tf.string, name='checkpoint_path'),
    ])
    def restore(self, checkpoint_path):
        path = checkpoint_path[0]
        results = tf.raw_ops.RestoreV2(
            prefix=path,
            tensor_names=tf.constant(['weight', 'bias']),
            shape_and_slices=tf.constant(['', '']),
            dtypes=[tf.float32, tf.float32],
        )
        self.w.assign(tf.reshape(results[0], [1, 1]))
        self.b.assign(tf.reshape(results[1], [1]))
        return {'status': tf.constant(0, dtype=tf.int32)}


def main():
    print(f'TensorFlow {tf.__version__}')

    model = LinearModelWithCheckpoint()

    # --- Export SavedModel ---
    saved_model_dir = '/tmp/flutter_litert_training_model_flex'
    print(f'Saving SavedModel to {saved_model_dir}...')
    tf.saved_model.save(
        model,
        saved_model_dir,
        signatures={
            'train': model.train,
            'infer': model.infer,
            'save': model.save,
            'restore': model.restore,
        },
    )

    # --- Convert to TFLite ---
    print('Converting to TFLite with SELECT_TF_OPS...')
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'test', 'assets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(
        os.path.join(out_dir, 'training_model_flex.tflite')
    )

    with open(out_path, 'wb') as f:
        f.write(tflite_model)

    print(f'Written {len(tflite_model):,} bytes -> {out_path}')

    # --- Verify signatures ---
    print('\nVerifying signatures...')
    interp = tf.lite.Interpreter(
        out_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO,
    )
    sigs = interp.get_signature_list()
    print(f'Signatures: {list(sigs.keys())}')
    for name, sig in sigs.items():
        print(f'  {name}: inputs={sig["inputs"]} outputs={sig["outputs"]}')

    expected = {'train', 'infer', 'save', 'restore'}
    if set(sigs.keys()) != expected:
        print(
            f'ERROR: expected signatures {expected}, got {set(sigs.keys())}',
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Smoke test: train and infer ---
    print('\nSmoke testing train/infer...')
    train_fn = interp.get_signature_runner('train')
    infer_fn = interp.get_signature_runner('infer')

    # Train a few steps.
    for _ in range(50):
        train_fn(
            x=np.array([[1.0]], dtype=np.float32),
            y=np.array([[2.0]], dtype=np.float32),
        )

    pred_trained = infer_fn(x=np.array([[1.0]], dtype=np.float32))
    print(f'  Prediction after 50 steps: {pred_trained["output"]}')
    assert pred_trained['output'][0][0] > 0.5, \
        f'Expected prediction > 0.5 after training, got {pred_trained["output"][0][0]}'

    # Note: save/restore via tf.raw_ops.Save/Restore may not work correctly
    # through the Python TFLite interpreter's Flex delegate. Full save/restore
    # testing is done in the Dart integration tests with the native FlexDelegate
    # library (test/native/flex_delegate_test.dart).
    print('  (save/restore tested in Dart integration tests)')

    print('\nSmoke tests passed.')
    print('Done.')


if __name__ == '__main__':
    main()
