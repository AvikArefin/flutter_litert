#!/usr/bin/env python3
"""
Generates test/assets/training_model.tflite

A minimal linear regression model (y = wx + b) with train, infer, get_weights,
and set_weights signatures. Uses only TFLITE_BUILTINS — no SELECT_TF_OPS required.

Requirements:
    pip install tensorflow>=2.13

Usage:
    python scripts/generate_training_model.py

Model properties:
    - train:       inputs x [1,1] float32, y [1,1] float32  -> loss [1] float32
    - infer:       input  x [1,1] float32                   -> output [1,1] float32
    - get_weights: no inputs -> w [1,1] float32, b [1] float32
    - set_weights: inputs w [1,1] float32, b [1] float32 -> w [1,1] float32, b [1] float32

Initial weights: w=0, b=0
  - infer(x=1)        -> 0.0 (deterministic for tests)
  - train(x=1, y=2)   -> loss=4.0 on first step ((0-2)^2=4)
  - After 50 steps    -> infer(x=1) converges toward 2.0
"""

import os
import sys
import tensorflow as tf


class LinearModel(tf.Module):
    def __init__(self):
        # Zero-initialized so test assertions are deterministic.
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

    @tf.function(input_signature=[])
    def get_weights(self):
        return {
            'w': self.w.read_value(),    # ReadVariable (builtin op 143)
            'b': self.b.read_value(),    # ReadVariable (builtin op 143)
        }

    @tf.function(input_signature=[
        tf.TensorSpec([1, 1], tf.float32, name='w'),
        tf.TensorSpec([1], tf.float32, name='b'),
    ])
    def set_weights(self, w, b):
        self.w.assign(w)    # AssignVariable (builtin op 144)
        self.b.assign(b)    # AssignVariable (builtin op 144)
        # Return read_value() so the assign ops have output consumers and
        # are not dead-code-eliminated by the TFLite converter.
        return {
            'w': self.w.read_value(),
            'b': self.b.read_value(),
        }


def main():
    print(f'TensorFlow {tf.__version__}')

    model = LinearModel()

    saved_model_dir = '/tmp/flutter_litert_training_model'
    print(f'Saving SavedModel to {saved_model_dir}...')
    tf.saved_model.save(
        model,
        saved_model_dir,
        signatures={
            'train': model.train,
            'infer': model.infer,
            'get_weights': model.get_weights,
            'set_weights': model.set_weights,
        },
    )

    print('Converting to TFLite...')
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'test', 'assets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, 'training_model.tflite'))

    with open(out_path, 'wb') as f:
        f.write(tflite_model)

    print(f'Written {len(tflite_model):,} bytes -> {out_path}')

    # Verify signatures using the TFLite interpreter.
    print('\nVerifying signatures...')
    interp = tf.lite.Interpreter(out_path)
    sigs = interp.get_signature_list()
    print(f'Signatures: {list(sigs.keys())}')
    for name, sig in sigs.items():
        print(f'  {name}: inputs={sig["inputs"]} outputs={sig["outputs"]}')

    expected = {'train', 'infer', 'get_weights', 'set_weights'}
    if set(sigs.keys()) != expected:
        print(f'ERROR: expected signatures {expected}, got {set(sigs.keys())}', file=sys.stderr)
        sys.exit(1)

    print('\nDone.')


if __name__ == '__main__':
    main()
