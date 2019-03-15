MAGRET_DIR="mid-corpus"
PREFIX="_mlm_split_magret"
ADJ_PREFIX="_mid_corpus"
SUFFIX="_mlm_mid"
VOCAB="global_vocab.csv" 
PREFIX_KERAS="keras${PREFIX}"
PREFIX_SKLEARN="sklearn${PREFIX}"
PREFIX_PYTORCH="pytorch${PREFIX}"

IN_KERAS=$MAGRET_DIR/${KERAS_PREFIX}_tk.txt
IN_SKLEARN=$MAGRET_DIR/${SKLEARN_PREFIX}_tk.txt
IN_PYTORCH=$MAGRET_DIR/${PYTORCH_PREFIX}_tk.txt

IN_VAL_KERAS=$MAGRET_DIR/${KERAS_PREFIX}_tk_val.txt
IN_VAL_SKLEARN=$MAGRET_DIR/${SKLEARN_PREFIX}_tk_val.txt
IN_VAL_PYTORCH=$MAGRET_DIR/${PYTORCH_PREFIX}_tk_val.txt

declare -a in_val_files=($IN_VAL_KERAS $IN_VAL_SKLEARN $IN_VAL_PYTORCH)
declare -a in_prefixes=($PREFIX_KERAS $PREFIX_SKLEARN $PREFIX_PYTORCH)
declare -a in_files=($IN_KERAS $IN_SKLEARN $IN_PYTORCH)

for prefix in "${in_prefixes[@]}"
do
  echo "$prefix"

  python prepare_pretraining_data.py \
    --input_file=$MAGRET_DIR/${prefix}_tk.txt \
    --output_file=$MAGRET_DIR/tf_examples${prefix}.tfrecord \
    --vocab_file=$MAGRET_DIR/$VOCAB \
    --adj_file=$MAGRET_DIR/adj/ \
    --do_lower_case=True \
    --max_seq_length=64 \
    --max_predictions_per_seq=1 \
    --masked_lm_prob=0.15 \
    --random_seed=1009 \
    --dupe_factor=50 \
    --sparse_adj=True \
    --adj_prefix=${prefix} \
    --is_training=True

  python prepare_pretraining_data.py \
    --input_file=$MAGRET_DIR/${prefix}_tk_val.txt \
    --output_file=$MAGRET_DIR/tf_examples_val${prefix}.tfrecord \
    --vocab_file=$MAGRET_DIR/$VOCAB \
    --adj_file=$MAGRET_DIR/adj/ \
    --do_lower_case=True \
    --max_seq_length=64 \
    --max_predictions_per_seq=1 \
    --masked_lm_prob=0.15 \
    --random_seed=1009 \
    --dupe_factor=50 \
    --sparse_adj=True \
    --adj_prefix=${prefix} \
    --is_training=False
done


