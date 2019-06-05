BIFOCALE_DIR="graph_cls"
PREFIX="MSRC_21"
PRETRAIN_DIR="graph_cls"

export CUDA_VISIBLE_DEVICES=0

python classifier.py \
  --do_train=True \
  --do_eval=False \
  --do_predict=True \
  --max_nb_preds=1009 \
  --task_name=methodname \
  --label_vocab=$BIFOCALE_DIR/${PREFIX}-vocab_label.txt \
  --vocab_file=$BIFOCALE_DIR/${PREFIX}-vocab.txt \
  --train_file=$BIFOCALE_DIR/${PREFIX}_cls_tk.txt \
  --train_labels=$BIFOCALE_DIR/${PREFIX}_label.txt \
  --train_adj=$BIFOCALE_DIR \
  --eval_file=$BIFOCALE_DIR/${PREFIX}_cls_tk_val.txt \
  --eval_labels=$BIFOCALE_DIR/${PREFIX}_label_val.txt \
  --eval_adj=$BIFOCALE_DIR \
  --data_dir=$BIFOCALE_DIR \
  --output_dir=$BIFOCALE_DIR/cls_output-no-lm-${PREFIX} \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=1e-4 \
  --num_train_epochs=120 \
  --save_checkpoints_steps=1000 \
  --bert_config_file=$PRETRAIN_DIR/shallow_config.json \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  --clean_data=True \
#  --init_checkpoint=$PRETRAIN_DIR/pretraining_output-${PREFIX}/model.ckpt-10000 \
#  --shuffle=True
