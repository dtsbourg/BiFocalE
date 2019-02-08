MAGRET_DIR="sparse"

python create_pretraining_data.py \
  --path=$MAGRET_DIR/data/ \
  --prefix=keras \
  --out_path=$MAGRET_DIR \
  --mode=varname \
  --pre=sparse_ \
  --nb_snippets=100000 \
  --sparse_adj 
