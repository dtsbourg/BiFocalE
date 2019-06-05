BIFOCALE_DIR="funcname_magret"
PREFIX="cls_with_funcsplit_magret"

python prepare_pretraining_data.py \
  --input_file=$BIFOCALE_DIR/${PREFIX}_tk_single.txt \
  --output_file=$BIFOCALE_DIR/tf_examples_single.tfrecord \
  --vocab_file=$BIFOCALE_DIR/vocab-code.txt \
  --adj_file=$BIFOCALE_DIR/${PREFIX}_adj_single.txt \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=50
