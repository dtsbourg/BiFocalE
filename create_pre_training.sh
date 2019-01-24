BERT_BASE_DIR="uncased_L-12_H-768_A-12"
CUSTOM_BERT_DIR="split_bert_with_multival"

python create_pretraining_data.py \
  --input_file=./split_bert_tk.txt \
  --output_file=$CUSTOM_BERT_DIR/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab-code.txt \
  --adj_file=./split_bert_adj.txt \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=50

python create_pretraining_data.py \
  --input_file=./split_bert_tk_val.txt \
  --output_file=$CUSTOM_BERT_DIR/tf_examples_val.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab-code.txt \
  --adj_file=./split_bert_adj_val.txt \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=50
