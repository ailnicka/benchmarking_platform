conda activate env-AE
cd scoring/data_sets_I
mkdir tani_eucl
echo 'Runnning for eucl/tani distance: Classical FPs'
python calculate_scored_lists.py -n 5 -f fps.txt -s Tanimoto -o tani_eucl
echo 'Runnning for eucl/tani distance: Compressed FPs'
python calculate_scored_lists.py -n 5 -f comp_fps.txt -s Euclidean -a -o tani_eucl
mkdir cos
echo 'Runnning for cos distance: Classical FPs'
python calculate_scored_lists.py -n 5 -f fps.txt -s Cosine_bit -o cos
echo 'Runnning for cos distance: Compressed FPs'
python calculate_scored_lists.py -n 5 -f comp_fps.txt -s Cosine -a -o cos
conda activate env-vs-py2
cd ../../validation/data_sets_I
mkdir tani_eucl
echo 'Runnning for eucl/tani validation'
python calculate_validation_methods.py -m methods.txt -i ../../scoring/data_sets_I/tani_eucl -o tani_eucl
mkdir cos
echo 'Runnning for cos validation'
python calculate_validation_methods.py -m methods.txt -i ../../scoring/data_sets_I/cos -o cos
cd ../../analysis/data_sets_I
mkdir tani_eucl
echo 'Runnning for eucl/tani analysis'
python run_analysis.py -i ../../validation/data_sets_I/tani_eucl -o tani_eucl
python run_method_summary.py -i tani_eucl
python run_fp_summary.py -i tani_eucl
mkdir cos
echo 'Runnning for cos analysis'
python run_analysis.py -i ../../validation/data_sets_I/cos -o cos
python run_method_summary.py -i cos
python run_fp_summary.py -i cos