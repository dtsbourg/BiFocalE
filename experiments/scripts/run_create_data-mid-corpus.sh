BIFOCALE_DIR="mid-corpus"
MODE="methodname"
SUFFIX="methodname_"


PREFIX_KERAS="keras"
PREFIX_SKLEARN="sklearn"
PREFIX_PYTORCH="pytorch"

declare -a prefixes=($PREFIX_KERAS $PREFIX_SKLEARN $PREFIX_PYTORCH)

for prefix in "${prefixes[@]}"
do
  echo "$prefix"

  python create_pretraining_data.py \
    --path=$BIFOCALE_DIR/data/ \
    --prefix=$prefix \
    --out_path=$BIFOCALE_DIR \
    --mode=$MODE \
    --pre=${prefix}_${SUFFIX} \
    --nb_snippets=100000 \
    --sparse_adj \
    --regen_vocab
done
