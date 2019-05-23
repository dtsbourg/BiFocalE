MAGRET_DIR="java-method"
PREFIX="java"
PRETRAIN_DIR="java"
PREDIR="methodname1"

export CUDA_VISIBLE_DEVICES=4

python classifier.py \
  --do_train=True \
  --do_eval=False \
  --do_predict=True \
  --max_nb_preds=1000 \
  --task_name=methodname \
  --label_vocab=$MAGRET_DIR/java-vocab-labels-thresh.txt \
  --vocab_file=$MAGRET_DIR/$PREDIR/java-vocab.txt \
  --train_file=$MAGRET_DIR/$PREDIR/${PREFIX}_tk.txt \
  --train_labels=$MAGRET_DIR/$PREDIR/java-train-labels.txt \
  --train_adj=$MAGRET_DIR/$PREDIR \
  --eval_file=$MAGRET_DIR/$PREDIR/${PREFIX}_tk_val.txt \
  --eval_labels=$MAGRET_DIR/$PREDIR/java-val-labels.txt \
  --eval_adj=$MAGRET_DIR/$PREDIR \
  --data_dir=$MAGRET_DIR \
  --output_dir=$MAGRET_DIR/cls_output-methodname-nopt5 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=1e-5 \
  --num_train_epochs=1000 \
  --save_checkpoints_steps=10000 \
  --bert_config_file=$MAGRET_DIR/shallow_config.json \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  --clean_data=True \
  #--init_checkpoint=$PRETRAIN_DIR/pretraining_output-java-400k/model.ckpt-400000 \
#  --shuffle=True
