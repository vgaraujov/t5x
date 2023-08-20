import functools
import seqio
import tensorflow as tf
import t5.data
from datasets import load_from_disk, load_dataset
from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics
from seqio import FunctionDataSource, utils

TaskRegistry = seqio.TaskRegistry
vocabulary = seqio.SentencePieceVocabulary('gs://t5-vlad-bucket/t5-data/spm.bpe.model', extra_ids=0)


DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=vocabulary, add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=vocabulary, add_eos=True)
}


seqio.TaskRegistry.add(
    'spanish_span_corruption',
    source=seqio.TFExampleDataSource(
        split_to_filepattern={
            'test': 'gs://t5-vlad-bucket/t5-data/valid.txt.tfrecords',
            'validation': 'gs://t5-vlad-bucket/t5-data/valid.txt.tfrecords',
            'train': 'gs://t5-vlad-bucket/t5-data/valid.txt.tfrecords',
        },
        feature_description={
            'text': tf.io.FixedLenFeature([], dtype=tf.string),
        }),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                'inputs': None,
                'targets': 'text'
            }),
        seqio.preprocessors.tokenize,
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])
