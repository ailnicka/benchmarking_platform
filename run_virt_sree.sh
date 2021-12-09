conda activate env-vs-py3
cd benchmarking_platform/scoring/data_sets_I
mkdir tani_eucl
python calculate_scored_lists.py -n 5 -f fps.txt -s Tanimoto -o tani_eucl
python calculate_scored_lists.py -n 5 -f comp_fps.txt -s Euclidean -a -o tani_eucl
mkdir cos
python calculate_scored_lists.py -n 5 -f fps.txt -s Cosine_bit -o cos
python calculate_scored_lists.py -n 5 -f comp_fps.txt -s Cosine -a -o cos
conda activate env-vs-py2
cd ../../validation/data_sets_I
mkdir tani_eucl
python calculate_validation_methods.py -m methods.txt -i ../../scoring/data_sets_I/tani_eucl -o tani_eucl
mkdir cos
python calculate_validation_methods.py -m methods.txt -i ../../scoring/data_sets_I/cos -o cos
cd ../../analysis/data_sets_I
mkdir tani_eucl
python run_analysis.py -i ../../validation/data_sets_I/tani_eucl -o tani_eucl
python run_method_summary.py -i tani_eucl
python run_fp_summary.py -i tani_eucl
mkdir cos
python run_analysis.py -i ../../validation/data_sets_I/cos -o cos
python run_method_summary.py -i cos
python run_fp_summary.py -i cos