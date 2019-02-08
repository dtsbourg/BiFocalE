MAGRET_DIR="sparse"
PREFIX="sparse_split_magret"
PRETRAIN_DIR="funcname_magret"

export CUDA_VISIBLE_DEVICES=0

python classifier.py \
  --do_train=True\
  --do_eval=False \
  --do_predict=True \
  --max_nb_preds=500 \
  --task_name=varname \
  --label_vocab=$MAGRET_DIR/vocab-code.txt \
  --vocab_file=$MAGRET_DIR/vocab-code.txt \
  --train_file=$MAGRET_DIR/${PREFIX}_tk.txt \
  --train_labels=$MAGRET_DIR/${PREFIX}_label.txt \
  --train_adj=$MAGRET_DIR \
  --eval_file=$MAGRET_DIR/${PREFIX}_tk_val.txt \
  --eval_labels=$MAGRET_DIR/${PREFIX}_label_val.txt \
  --eval_adj=$MAGRET_DIR \
  --data_dir=$MAGRET_DIR \
  --output_dir=$MAGRET_DIR/cls_output \
  --max_seq_length=64 \
  --train_batch_size=16 \
  --learning_rate=2e-6 \
  --num_train_epochs=1 \
  --save_checkpoints_steps=500 \
  --init_checkpoint=$PRETRAIN_DIR/pretraining_output/model.ckpt-200000 \
  --bert_config_file=$PRETRAIN_DIR/bert_config.json \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  #--clean_data=True
