#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:57:29 2019

@author: subhajit
"""

import collections
import tensorflow as tf
import numpy as np
import run_squad, modeling, tokenization



def make_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case):
  
  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = run_squad._get_best_indexes(result.start_logits, n_best_size)
      end_indexes = run_squad._get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      feature_null_score = result.start_logits[0] + result.end_logits[0]
      if feature_null_score < score_null:
        score_null = feature_null_score
        min_null_feature_index = feature_index
        null_start_logit = result.start_logits[0]
        null_end_logit = result.end_logits[0]
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index]))

    prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = run_squad.get_final_text(tok_text, orig_text, do_lower_case)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if "" not in seen_predictions:
      nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = run_squad._compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    
    score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
    scores_diff_json[example.qas_id] = score_diff
    if score_diff > 0:
      all_predictions[example.qas_id] = ""
    else:
      all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json
  
  return all_predictions, all_nbest_json, scores_diff_json




########### Initialization: Change the values before running ###########
tf.reset_default_graph()
bert_config = modeling.BertConfig.from_json_file('~/BERT/uncased_L-12_H-768_A-12/bert_config.json')
init_checkpoint = '~/bert-master/outdir/'
vocab_file = '~/BERT/uncased_L-12_H-768_A-12/vocab.txt'
predict_file = '~/bert-master/example.json'
#output_dir = '~/bert-master/outdir2'
seq_length = 128
doc_stride = 128
query_length = 64
use_tpu = False
n_best_size = 20
max_answer_length = 30
do_lower_case = True




########### Feature Creation ###########
eval_examples = run_squad.read_squad_examples(input_file=predict_file, is_training=False)

#eval_writer = run_squad.FeatureWriter(
#    filename=os.path.join(output_dir, "eval.tf_record"),
#    is_training=False)
eval_features = []

def append_feature(feature):
  eval_features.append(feature)
  #eval_writer.process_feature(feature)

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

run_squad.convert_examples_to_features(
    examples=eval_examples,
    tokenizer=tokenizer,
    max_seq_length=seq_length,
    doc_stride=doc_stride,
    max_query_length=query_length,
    is_training=False,
    output_fn=append_feature)




########### Re-load model from saved checkpoint ###########
#unique_ids = tf.placeholder([], tf.int64)
input_ids = tf.placeholder(tf.int64, [None, seq_length])
input_mask = tf.placeholder(tf.int64, [None, seq_length])
segment_ids = tf.placeholder(tf.int64, [None, seq_length])

(start_logits, end_logits) = run_squad.create_model(
    bert_config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    use_one_hot_embeddings=use_tpu)

saver = tf.train.Saver()

with tf.Session() as sess:
    #saver = tf.train.import_meta_graph(init_checkpoint+'model.ckpt-16289.meta')
    saver.restore(sess, tf.train.latest_checkpoint(init_checkpoint))
    
    preds = []
    for i in range(len(eval_features)):
        result = sess.run([start_logits, end_logits] ,feed_dict={input_ids: np.array(eval_features[i].input_ids).reshape(1,-1),
                                                        input_mask: np.array(eval_features[i].input_mask).reshape(1,-1),
                                                        segment_ids: np.array(eval_features[i].segment_ids).reshape(1,-1)})
        preds.append({"unique_ids": eval_features[i].unique_id, "start_logits": result[0], "end_logits": result[1]})




########### Prediction ###########
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])
all_results = []
for result in preds:
    unique_id = int(result["unique_ids"])
    start_logits = [float(x) for x in result["start_logits"].flat]
    end_logits = [float(x) for x in result["end_logits"].flat]
    all_results.append(
        RawResult(
            unique_id=unique_id,
            start_logits=start_logits,
            end_logits=end_logits))

all_predictions, all_nbest_json, scores_diff_json = make_predictions(eval_examples, eval_features, all_results,
                                                                      n_best_size, max_answer_length,
                                                                      do_lower_case)










########### Visualization ###########
#import tensorflow as tf
#from tensorflow.python.summary.writer.writer import FileWriter
#FileWriter('/tmp/tensorflow_logdir', graph=tf.get_default_graph()).close()
#tensorboard --logdir=/tmp/tensorflow_logdir
