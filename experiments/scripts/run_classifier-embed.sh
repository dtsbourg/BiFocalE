BIFOCALE_DIR="sparse"
PREFIX="sparse_fname2_split_magret"
PRETRAIN_DIR="sparse"

export CUDA_VISIBLE_DEVICES=1

python classifier.py \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --do_embed=True \
  --max_nb_preds=600 \
  --task_name=methodname \
  --label_vocab=$BIFOCALE_DIR/sparse_fname2_vocab-label.txt \
  --vocab_file=$BIFOCALE_DIR/sparse_tmp_vocab-code.txt \
  --train_file=$BIFOCALE_DIR/${PREFIX}_tk.txt \
  --train_labels=$BIFOCALE_DIR/${PREFIX}_label.txt \
  --train_adj=$BIFOCALE_DIR \
  --eval_file=$BIFOCALE_DIR/${PREFIX}_tk_val.txt \
  --eval_labels=$BIFOCALE_DIR/${PREFIX}_label_val.txt \
  --eval_adj=$BIFOCALE_DIR \
  --data_dir=$BIFOCALE_DIR \
  --output_dir=$BIFOCALE_DIR/cls_output-embed \
  --max_seq_length=64 \
  --train_batch_size=16 \
  --learning_rate=1e-6 \
  --num_train_epochs=200 \
  --save_checkpoints_steps=500 \
  --init_checkpoint=$PRETRAIN_DIR/pretraining_output-200k/model.ckpt-200000 \
  --bert_config_file=$PRETRAIN_DIR/bert_config.json \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  --clean_data=True \
#  --shuffle=True
