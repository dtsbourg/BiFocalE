BIFOCALE_DIR="graph_cls"
PREFIX="MSRC_9"
SUFFIX="_MSRC_9"
#PREFIX_KERAS="keras_mlm_split_magret_"
#PREFIX_SKLEARN="sklearn_mlm_split_magret_"

python prepare_pretraining_data.py \
  --input_file=$BIFOCALE_DIR/${PREFIX}_tk.txt \
  --output_file=$BIFOCALE_DIR/tf_examples${SUFFIX}.tfrecord \
  --vocab_file=$BIFOCALE_DIR/${PREFIX}-vocab.txt \
  --adj_file=$BIFOCALE_DIR/adj/ \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=50 \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  --is_training=True

python prepare_pretraining_data.py \
  --input_file=$BIFOCALE_DIR/${PREFIX}_tk_val.txt \
  --output_file=$BIFOCALE_DIR/tf_examples_val${SUFFIX}.tfrecord \
  --vocab_file=$BIFOCALE_DIR/${PREFIX}-vocab.txt \
  --adj_file=$BIFOCALE_DIR/adj/ \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=50 \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  --is_training=False
