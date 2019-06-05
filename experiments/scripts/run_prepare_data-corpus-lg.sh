BIFOCALE_DIR="large-corpus"
PREFIX="_mlm_split_magret"
ADJ_PREFIX="_lg_corpus"
SUFFIX="_mlm_large"
#VOCAB="sparse_tmp_vocab-code.txt"
VOCAB="global_vocab.csv" 
PREFIX_KERAS="keras${PREFIX}"
PREFIX_SKLEARN="sklearn${PREFIX}"
PREFIX_PYTORCH="pytorch${PREFIX}"
PREFIX_ANSIBLE="ansible${PREFIX}"
PREFIX_REQUESTS="requests${PREFIX}"
PREFIX_DJANGO="django${PREFIX}"
PREFIX_HTTPIE="httpie${PREFIX}"
PREFIX_YT="youtube-dl${PREFIX}"
PREFIX_FLASK="flask${PREFIX}"
PREFIX_BERT="bert${PREFIX}"


IN_KERAS=$BIFOCALE_DIR/${KERAS_PREFIX}_tk.txt
IN_SKLEARN=$BIFOCALE_DIR/${SKLEARN_PREFIX}_tk.txt
IN_PYTORCH=$BIFOCALE_DIR/${PYTORCH_PREFIX}_tk.txt
IN_ANSIBLE=$BIFOCALE_DIR/${ANSIBLE_PREFIX}_tk.txt
IN_REQUESTS=$BIFOCALE_DIR/${REQUESTS_PREFIX}_tk.txt
IN_DJANGO=$BIFOCALE_DIR/${DJANGO_PREFIX}_tk.txt
IN_HTTPIE=$BIFOCALE_DIR/${HTTPIE_PREFIX}_tk.txt
IN_YT=$BIFOCALE_DIR/${YT_PREFIX}_tk.txt
IN_FLASK=$BIFOCALE_DIR/${FLASK_PREFIX}_tk.txt
IN_BERT=$BIFOCALE_DIR/${BERT_PREFIX}_tk.txt

IN_VAL_KERAS=$BIFOCALE_DIR/${KERAS_PREFIX}_tk_val.txt
IN_VAL_SKLEARN=$BIFOCALE_DIR/${SKLEARN_PREFIX}_tk_val.txt
IN_VAL_PYTORCH=$BIFOCALE_DIR/${PYTORCH_PREFIX}_tk_val.txt
IN_VAL_ANSIBLE=$BIFOCALE_DIR/${ANSIBLE_PREFIX}_tk_val.txt
IN_VAL_REQUESTS=$BIFOCALE_DIR/${REQUESTS_PREFIX}_tk_val.txt
IN_VAL_DJANGO=$BIFOCALE_DIR/${DJANGO_PREFIX}_tk_val.txt
IN_VAL_HTTPIE=$BIFOCALE_DIR/${HTTPIE_PREFIX}_tk_val.txt
IN_VAL_YT=$BIFOCALE_DIR/${YT_PREFIX}_tk_val.txt
IN_VAL_FLASK=$BIFOCALE_DIR/${FLASK_PREFIX}_tk_val.txt
IN_VAL_BERT=$BIFOCALE_DIR/${BERT_PREFIX}_tk_val.txt

declare -a in_val_files=($IN_VAL_KERAS $IN_VAL_SKLEARN $IN_VAL_PYTORCH $IN_VAL_ANSIBLE $IN_VAL_REQUESTS $IN_VAL_DJANGO $IN_VAL_HTTPIE $IN_VAL_YT $IN_VAL_FLASK $IN_VAL_BERT)
#declare -a in_prefixes=($PREFIX_KERAS $PREFIX_SKLEARN $PREFIX_PYTORCH $PREFIX_ANSIBLE $PREFIX_REQUESTS $PREFIX_DJANGO $PREFIX_HTTPIE $PREFIX_YT $PREFIX_FLASK $PREFIX_BERT)
declare -a in_prefixes=($PREFIX_YT $PREFIX_FLASK)
declare -a in_files=($IN_KERAS $IN_SKLEARN $IN_PYTORCH $IN_ANSIBLE $IN_REQUESTS $IN_DJANGO $IN_HTTPIE $IN_YT $IN_FLASK $IN_BERT)

for prefix in "${in_prefixes[@]}"
do
  echo "$prefix"

  python prepare_pretraining_data.py \
    --input_file=$BIFOCALE_DIR/${prefix}_tk.txt \
    --output_file=$BIFOCALE_DIR/tf_examples${prefix}.tfrecord \
    --vocab_file=$BIFOCALE_DIR/$VOCAB \
    --adj_file=$BIFOCALE_DIR/adj/ \
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
    --input_file=$BIFOCALE_DIR/${prefix}_tk_val.txt \
    --output_file=$BIFOCALE_DIR/tf_examples_val${prefix}.tfrecord \
    --vocab_file=$BIFOCALE_DIR/$VOCAB \
    --adj_file=$BIFOCALE_DIR/adj/ \
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


