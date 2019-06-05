BIFOCALE_DIR="node_cls"
PREFIX="CORA"
SUFFIX="_CORA"

python prepare_pretraining_data.py \
  --input_file=$BIFOCALE_DIR/${PREFIX}_tk.txt \
  --output_file=$BIFOCALE_DIR/tf_examples${SUFFIX}.tfrecord \
  --vocab_file=$BIFOCALE_DIR/${PREFIX}-vocab.txt \
  --adj_file=$BIFOCALE_DIR/adj/ \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=1 \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  --is_training=True

python prepare_pretraining_data.py \
  --input_file=$BIFOCALE_DIR/${PREFIX}_tk_val.txt \
  --output_file=$BIFOCALE_DIR/tf_examples_val${SUFFIX}.tfrecord \
  --vocab_file=$BIFOCALE_DIR/${PREFIX}-vocab.txt \
  --adj_file=$BIFOCALE_DIR/adj/ \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=1 \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  --is_training=False
