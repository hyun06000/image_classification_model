echo Warning : The existed TFRs will be removed.

rm ./data/TFRs/train/*
rm ./data/TFRs/validation/*

rm log_make_TFRs.out

python data_preparation.py > log_make_TFRs.out

echo -ne "\007"
