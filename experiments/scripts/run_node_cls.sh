BIFOCALE_DIR="node_cls"
PREFIX="CORA"
export CUDA_VISIBLE_DEVICES=2

python run_pretraining.py \
  --input_file=$BIFOCALE_DIR/tf_examples_${PREFIX}.tfrecord \
  --validation_file=$BIFOCALE_DIR/tf_examples_val_${PREFIX}.tfrecord \
  --output_dir=$BIFOCALE_DIR/pretraining_output-${PREFIX} \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --save_prediction=True \
  --save_attention=False \
  --bert_config_file=$BIFOCALE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --num_train_steps=2000 \
  --save_checkpoints_steps=500 \
  --num_warmup_steps=10 \
  --learning_rate=1e-5
