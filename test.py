from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import coref_model as cm
import util

if __name__ == "__main__":
  config = util.initialize_from_env()

  model = cm.CorefModel(config)

  input_filename = 'dev.english.jsonlines'

  with tf.Session() as session:
    model.restore(session)

    with open(input_filename) as input_file:
      for example_num, line in enumerate(input_file.readlines()):
        example = json.loads(line)
        tensorized_example = model.tensorize_example(example, is_training=False)
        feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
        _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
        print('top_antecedents', top_antecedents)
        predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
        print('antecedents', predicted_antecedents)
        example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
        print('clusters', example['predicted_clusters'])
        break
