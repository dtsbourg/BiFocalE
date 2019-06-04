MAGRET_DIR="graph_cls"
PREFIX="MSRC_21"
PRETRAIN_DIR="graph_cls"

export CUDA_VISIBLE_DEVICES=0

python classifier.py \
  --do_train=True \
  --do_eval=False \
  --do_predict=True \
  --max_nb_preds=1009 \
  --task_name=methodname \
  --label_vocab=$MAGRET_DIR/${PREFIX}-vocab_label.txt \
  --vocab_file=$MAGRET_DIR/${PREFIX}-vocab.txt \
  --train_file=$MAGRET_DIR/${PREFIX}_cls_tk.txt \
  --train_labels=$MAGRET_DIR/${PREFIX}_label.txt \
  --train_adj=$MAGRET_DIR \
  --eval_file=$MAGRET_DIR/${PREFIX}_cls_tk_val.txt \
  --eval_labels=$MAGRET_DIR/${PREFIX}_label_val.txt \
  --eval_adj=$MAGRET_DIR \
  --data_dir=$MAGRET_DIR \
  --output_dir=$MAGRET_DIR/cls_output-no-lm-${PREFIX} \
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
