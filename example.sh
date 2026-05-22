python main_reweight.py --dataset="Twitter" --model="LightGCN" \
--save=1 --neg_in_val_test=1 --seed=1 \
--epochs=70 --save_path="/log_1_1.2.txt" --penalty=1.2


python test_model.py --dataset="Twitter" --model="LightGCN" --seed=1 --reweight_flag=2 --penalty=1.4


python main.py --dataset="Twitter" --model="LightGCN" \
--save=1 --neg_in_val_test=1 --seed=1 \
--epochs=70 --save_path="/log_without_reweight.txt"

python main_reweight5.py --dataset="Twitter" --model="LightGCN" \
--save=1 --neg_in_val_test=1 --seed=1 \
--epochs=70 --save_path="/log_5_1.0.txt" --penalty=1.0


python main_mf_reweight5.py --dataset="Twitter" --model="MF" --save=1 --neg_in_val_test=1 --seed=1 --epochs=50 --save_path="/log_mf_rw_1.2.txt" --penalty=1.2 --reweight_flag=1

python main_mf.py --dataset="Twitter" --model="MF" --save=1 --neg_in_val_test=1 --seed=1 --epochs=50 --save_path="/log_mf.txt" 



python test_MF.py --dataset="Twitter" --model="MF" --seed=1 --reweight_flag=2 --penalty=1.2

python main_neumf_reweight5.py --dataset="Twitter" --model="NeuMF" --save=1 --neg_in_val_test=1 --seed=1 --epochs=50 --save_path="/log_neumf_rw_1.2.txt" --penalty=1.2 --reweight_flag=1

python main_neumf.py --dataset="Twitter" --model="NeuMF" --save=1 --neg_in_val_test=1 --seed=1 --epochs=50 --save_path="/log_neumf.txt" 

python degree_debias_rerank.py \
  --dataset Twitter \
  --seed 1 \
  --method MF \
  --eval_flag test \
  --alphas 0,0.1,0.2,0.3,0.5,0.8,1.0 \
  --base_dir .

python rerank_parrel.py



python test_NeuMF.py --dataset="Twitter" --model="NeuMF" --seed=1 --reweight_flag=2 --penalty=1.2
