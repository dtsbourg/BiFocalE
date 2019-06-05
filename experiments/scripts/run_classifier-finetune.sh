BIFOCALE_DIR="large-corpus"
PREFIX="sparse_varname_split_magret"
PREFIX_PYTORCH="pytorch_cls_split_magret"
PREFIX_KERAS="keras_cls_split_magret"
PREFIX_SKLEARN="sklearn_cls_split_magret"
PRETRAIN_DIR="large-corpus"

export CUDA_VISIBLE_DEVICES=1

python classifier.py \
  --do_train=True \
  --do_eval=False \
  --do_predict=True \
  --max_nb_preds=600 \
  --task_name=methodname \
  --label_vocab=$BIFOCALE_DIR/label_vocab.csv \
  --vocab_file=$BIFOCALE_DIR/global_vocab.csv \
  --train_file=$BIFOCALE_DIR/${PREFIX_KERAS}_tk.txt \
  --train_labels=$BIFOCALE_DIR/${PREFIX_KERAS}_label.txt \
  --train_adj=$BIFOCALE_DIR \
  --eval_file=$BIFOCALE_DIR/${PREFIX_KERAS}_tk_val.txt \
  --eval_labels=$BIFOCALE_DIR/${PREFIX_KERAS}_label_val.txt \
  --eval_adj=$BIFOCALE_DIR \
  --data_dir=$BIFOCALE_DIR \
  --output_dir=$BIFOCALE_DIR/cls_output-finetune \
  --max_seq_length=64 \
  --train_batch_size=16 \
  --learning_rate=1e-6 \
  --num_train_epochs=50 \
  --save_checkpoints_steps=500 \
  --init_checkpoint=$PRETRAIN_DIR/pretraining_output-200k/model.ckpt-200000 \
  --bert_config_file=$PRETRAIN_DIR/large_config.json \
  --sparse_adj=True \
  --adj_prefix=${PREFIX_KERAS} \
   --clean_data=True \
#  --shuffle=True
