MAGRET_DIR="cls_magret"

python create_pretraining_data.py \
  --path=$MAGRET_DIR/data/ \
  --prefix=keras \
  --out_path=$MAGRET_DIR \
  --nb_snippets=100000
