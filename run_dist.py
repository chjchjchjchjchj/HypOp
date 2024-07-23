from src.run_exp import  exp_centralized, exp_centralized_for, exp_centralized_for_multi
from src.solver import QUBO_solver
import json
import torch.multiprocessing as mp


if __name__ == '__main__':  
   test_mode = "dist"
#    test_mode = "infer"
   dataset = "stanford"

   if test_mode == "infer":
      if dataset == "stanford":
        #  with open('infer_configs/maxcut_R_for.json') as f:
         with open('dist_configs/maxind_stanford_p1.json') as f:
            params = json.load(f)
      elif dataset == "arxiv":
         with open('infer_configs/maxcut_arxiv_for.json') as f:
            params = json.load(f)
      exp_centralized(params)

   elif test_mode == "dist":
      if dataset == "stanford":
        #  with open('dist_configs/maxcut_R_for.json') as f:
         with open('dist_configs/maxind_stanford_p1.json') as f:
            params = json.load(f)
      elif dataset == "arxiv":
         with open('dist_configs/maxcut_arxiv_for.json') as f:
            params = json.load(f)
      
      params["logging_path"] = params["logging_path"].split(".log")[0] +str(params["multi_gpu"]) + "_" + params["data"] + "_test.log"
      if params["multi_gpu"]:
         params['num_samples'] = 1
        #  print(f"params={params}")
        #  import sys; sys.exit()
        #  params={'data': 'stanford', 'mode': 'maxcut', 'K': 1, 'random_init': 'none', 'transfer': False, 'initial_transfer': False, 'GD': False, 'lr': 0.01, 'epoch': 100.0, 'tol': 0.0001, 'mapping': 'distribution', 'boosting_mapping': 1, 'logging_path': './log/maxcut_R_for1_stanford_test.log', 'res_path': './res/maxcut_R_for.hdf5', 'folder_path': './data/maxcut_data/stanford_data_single/', 'plot_path': './res/plots/Hist_maxcut/', 'model_save_path': './models/maxcut/', 'model_load_path': './models/maxcut/', 'N_realize': 1, 'Niter_h': 0, 't': 0.5, 'hyper': False, 'penalty': 0, 'patience': 50, 'multi_gpu': 1, 'num_gpus': 4, 'test_multi_gpu': 0, 'load_best_out': False, 'coarsen': False, 'f_input': False, 'sparcify': False, 'sparcify_p': 0.8, 'n_partitions': 4, 'Att': False, 'dropout': 0, 'num_samples': 1}
         mp.spawn(exp_centralized_for_multi, args=(list(range(params["num_gpus"])), params), nprocs=params["num_gpus"])
      else:
         exp_centralized_for(params)