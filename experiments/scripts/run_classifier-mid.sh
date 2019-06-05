BIFOCALE_DIR="mid-corpus"


PREFIX="_methodname_split_magret"
ADJ_PREFIX="_mid_corpus"
SUFFIX="_methodname_mid"

KERAS_PREFIX="keras${PREFIX}"
SKLEARN_PREFIX="sklearn${PREFIX}"
#PYTORCH_PREFIX="pytorch${PREFIX}"

IN_KERAS=$BIFOCALE_DIR/${KERAS_PREFIX}_tk.txt
IN_SKLEARN=$BIFOCALE_DIR/${SKLEARN_PREFIX}_tk.txt
#IN_PYTORCH=$BIFOCALE_DIR/${PYTORCH_PREFIX}_tk.txt

IN_LABEL_KERAS=$BIFOCALE_DIR/${KERAS_PREFIX}_label.txt
IN_LABEL_SKLEARN=$BIFOCALE_DIR/${SKLEARN_PREFIX}_label.txt
#IN_LABEL_PYTORCH=$BIFOCALE_DIR/${PYTORCH_PREFIX}_label.txt

IN_VAL_KERAS=$BIFOCALE_DIR/${KERAS_PREFIX}_tk_val.txt
IN_VAL_SKLEARN=$BIFOCALE_DIR/${SKLEARN_PREFIX}_tk_val.txt
#IN_VAL_PYTORCH=$BIFOCALE_DIR/${PYTORCH_PREFIX}_tk_val.txt

IN_LABEL_VAL_KERAS=$BIFOCALE_DIR/${KERAS_PREFIX}_label_val.txt
IN_LABEL_VAL_SKLEARN=$BIFOCALE_DIR/${SKLEARN_PREFIX}_label_val.txt
#IN_LABEL_VAL_PYTORCH=$BIFOCALE_DIR/${PYTORCH_PREFIX}_label_val.txt


PRETRAIN_DIR="mid-corpus"

export CUDA_VISIBLE_DEVICES=0

python classifier.py \
  --do_train=True \
  --do_eval=False \
  --do_predict=True \
  --max_nb_preds=100000 \
  --task_name=methodname \
  --init_checkpoint=$PRETRAIN_DIR/pretraining_output-300k/model.ckpt-500000 \
  --label_vocab=$BIFOCALE_DIR/label_vocab.csv \
  --vocab_file=$BIFOCALE_DIR/global_vocab.csv \
  --train_file=$IN_KERAS,$IN_SKLEARN \
  --train_labels=$IN_LABEL_KERAS,$IN_LABEL_SKLEARN \
  --train_adj=$BIFOCALE_DIR \
  --eval_file=$IN_VAL_KERAS,$IN_VAL_SKLEARN \
  --eval_labels=$IN_LABEL_VAL_KERAS,$IN_LABEL_VAL_SKLEARN \
  --eval_adj=$BIFOCALE_DIR \
  --data_dir=$BIFOCALE_DIR \
  --output_dir=$BIFOCALE_DIR/cls_output-methodname-2-nopt \
  --max_seq_length=64 \
  --train_batch_size=16 \
  --learning_rate=1e-4 \
  --num_train_epochs=200 \
  --save_checkpoints_steps=10000 \
  --bert_config_file=$PRETRAIN_DIR/shallow_config.json \
  --sparse_adj=True \
  --adj_prefix=$KERAS_PREFIX,$SKLEARN_PREFIX \
  --clean_data=True \
#  --shuffle=True
  #--train_file=$IN_KERAS,$IN_SKLEARN,$IN_PYTORCH \


