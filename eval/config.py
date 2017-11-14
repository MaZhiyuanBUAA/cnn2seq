#evaluation configuration
path = {
	'pred_path':'/home/zyma/work/models/gans/result.txt',
	'target_path':'/home/zyma/work/models/seq2BF/results/results_3_128_110000',
	'emb_path':'/home/zyma/work/models/algr_opt/nn_models/embeddings.pkl',
	'vocab_path':'/home/zyma/work/models/gans/gen_data/vocab110000.all',
	'idf_path':'daily_idf.pkl',
	'post_path':'std_test.post'
	}

distinct_config = [{'test_size':100,'test_times':50},
	{'test_size':50,'test_times':50}]

model_result = [{'model':'TG','result_path':'/home/zyma/work/models/target_glimpse/results/results.txt'},
		{'model':'baseline','result_path':'/home/zyma/work/models/algr_opt/results/results_bl'},
                {'model':'gans_11w','result_path':'/home/zyma/work/models/gans/results.txt'},
		{'model':'gans_2.5w','result_path':'/home/zyma/work/models/gans/results_25000.txt'},
		{'model':'BLBS','result_path':'/home/zyma/work/models/algr_opt/results/result_beamSearch'},
                {'model':'Seq2BF','result_path':'/home/zyma/work/models/seq2BF/results/results_3_128_110000'}]
