MAGRET_DIR="large-corpus"
PREFIX="flask_mlm_split_magret"
SUFFIX="flask_mlm_split_magret"
#VOCAB="sparse_tmp_vocab-code.txt"
VOCAB="global_vocab.csv" 
#PREFIX_KERAS="keras_mlm_split_magret_"
#PREFIX_SKLEARN="sklearn_mlm_split_magret_"

python prepare_pretraining_data.py \
  --input_file=$MAGRET_DIR/${PREFIX}_tk.txt \
  --output_file=$MAGRET_DIR/tf_examples${SUFFIX}.tfrecord \
  --vocab_file=$MAGRET_DIR/$VOCAB \
  --adj_file=$MAGRET_DIR/adj/ \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=50 \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  --is_training=True

python prepare_pretraining_data.py \
  --input_file=$MAGRET_DIR/${PREFIX}_tk_val.txt \
  --output_file=$MAGRET_DIR/tf_examples_val${SUFFIX}.tfrecord \
  --vocab_file=$MAGRET_DIR/$VOCAB \
  --adj_file=$MAGRET_DIR/adj/ \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=50 \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  --is_training=False
