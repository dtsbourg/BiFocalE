MAGRET_DIR="cls_magret"
PREFIX="clssplit_magret"

python prepare_pretraining_data.py \
  --input_file=$MAGRET_DIR/${PREFIX}_tk.txt \
  --output_file=$MAGRET_DIR/tf_examples.tfrecord \
  --vocab_file=$MAGRET_DIR/vocab-code.txt \
  --adj_file=$MAGRET_DIR/${PREFIX}_adj.txt \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=50

python prepare_pretraining_data.py \
  --input_file=$MAGRET_DIR/${PREFIX}_tk_val.txt \
  --output_file=$MAGRET_DIR/tf_examples_val.tfrecord \
  --vocab_file=$MAGRET_DIR/vocab-code.txt \
  --adj_file=$MAGRET_DIR/${PREFIX}_adj_val.txt \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=50

