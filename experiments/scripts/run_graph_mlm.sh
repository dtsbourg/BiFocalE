MAGRET_DIR="graph_cls"
export CUDA_VISIBLE_DEVICES=2

python run_pretraining.py \
  --input_file=$MAGRET_DIR/tf_examples_Tox21_AHR.tfrecord \
  --validation_file=$MAGRET_DIR/tf_examples_val_Tox21_AHR.tfrecord \
  --output_dir=$MAGRET_DIR/pretraining_output-Tox21_AHR \
  --do_train=True \
  --do_eval=True \
  --do_predict=False \
  --save_prediction=False \
  --save_attention=False \
  --bert_config_file=$MAGRET_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --num_train_steps=10000 \
  --save_checkpoints_steps=500 \
  --num_warmup_steps=10 \
  --learning_rate=1e-5
