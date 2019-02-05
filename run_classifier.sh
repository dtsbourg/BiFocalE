MAGRET_DIR="funcname_magret"
PREFIX="cls_funcdefsplit_magret"

export CUDA_VISIBLE_DEVICES=0

python classifier.py \
  --do_train=True\
  --do_eval=True \
  --do_predict=True \
  --task_name=methodname \
  --label_vocab=$MAGRET_DIR/vocab-label.txt \
  --vocab_file=$MAGRET_DIR/vocab-code.txt \
  --train_file=$MAGRET_DIR/${PREFIX}_tk.txt \
  --train_labels=$MAGRET_DIR/${PREFIX}_label.txt \
  --train_adj=$MAGRET_DIR/${PREFIX}_adj.txt \
  --eval_file=$MAGRET_DIR/${PREFIX}_tk_val.txt \
  --eval_labels=$MAGRET_DIR/${PREFIX}_label_val.txt \
  --eval_adj=$MAGRET_DIR/${PREFIX}_adj_val.txt \
  --data_dir=$MAGRET_DIR \
  --output_dir=$MAGRET_DIR/cls_output \
  --max_seq_length=64 \
  --train_batch_size=16 \
  --learning_rate=2e-6 \
  --num_train_epochs=2000 \
  --init_checkpoint=$MAGRET_DIR/pretraining_output/model.ckpt-200000 \
  --bert_config_file=$MAGRET_DIR/bert_config.json
