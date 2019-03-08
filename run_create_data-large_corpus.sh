MAGRET_DIR="large-corpus"
MODE="methodname"
SUFFIX="methodname_"


PREFIX_KERAS="keras"
PREFIX_SKLEARN="sklearn"
PREFIX_PYTORCH="pytorch"
PREFIX_ANSIBLE="ansible"
PREFIX_REQUESTS="requests"
PREFIX_DJANGO="django"
PREFIX_HTTPIE="httpie"
PREFIX_YT="youtube-dl"
PREFIX_FLASK="flask"
PREFIX_BERT="bert"

declare -a prefixes=($PREFIX_KERAS $PREFIX_SKLEARN $PREFIX_PYTORCH $PREFIX_ANSIBLE $PREFIX_REQUESTS $PREFIX_DJANGO $PREFIX_HTTPIE $PREFIX_YT $PREFIX_FLASK $PREFIX_BERT)

for prefix in "${prefixes[@]}"
do
  echo "$prefix"

  python create_pretraining_data.py \
    --path=$MAGRET_DIR/data/ \
    --prefix=$prefix \
    --out_path=$MAGRET_DIR \
    --mode=$MODE \
    --pre=${prefix}_${SUFFIX} \
    --nb_snippets=100000 \
    --sparse_adj \
    --regen_vocab
done
