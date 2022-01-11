conda activate env-AE
now=$(date +"%Y_%m_%d_%H_%M")
cd scoring/data_sets_I
mkdir tani_eucl_$now
echo 'Runnning for eucl/tani distance: Classical FPs'
python calculate_scored_lists.py -n 5 -f bench_comp_fps_bin.txt -s Tanimoto -o tani_eucl_$now
echo 'Runnning for eucl/tani distance: Compressed FPs'
python calculate_scored_lists.py -n 5 -f bench_comp_fps.txt -s Euclidean -a -o tani_eucl_$now
mkdir cos_$now
echo 'Runnning for cos distance: Classical FPs'
python calculate_scored_lists.py -n 5 -f bench_comp_fps_bin.txt -s Cosine_bit -o cos_$now
echo 'Runnning for cos distance: Compressed FPs'
python calculate_scored_lists.py -n 5 -f bench_comp_fps.txt -s Cosine -a -o cos_$now
conda activate env-vs-py2
cd ../../validation/data_sets_I
mkdir tani_eucl_$now
echo 'Runnning for eucl/tani validation'
python calculate_validation_methods.py -m methods.txt -i ../../scoring/data_sets_I/tani_eucl_$now -o tani_eucl_$now
mkdir cos_$now
echo 'Runnning for cos validation'
python calculate_validation_methods.py -m methods.txt -i ../../scoring/data_sets_I/cos_$now -o cos_$now
cd ../../analysis/data_sets_I
mkdir tani_eucl_$now
echo 'Runnning for eucl/tani analysis'
python run_analysis.py -i ../../validation/data_sets_I/tani_eucl_$now -o tani_eucl_$now
python run_method_summary.py -i tani_eucl_$now
python run_fp_summary.py -i tani_eucl_$now
mkdir cos_$now
echo 'Runnning for cos analysis'
python run_analysis.py -i ../../validation/data_sets_I/cos_$now -o cos_$now
python run_method_summary.py -i cos_$now
python run_fp_summary.py -i cos_$now