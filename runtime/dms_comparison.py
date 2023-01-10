
import json
import numpy as np
import pandas as pd
from scipy.special import softmax

from protein_holography_pytorch.utils.protein_naming import *

import argparse
from protein_holography_pytorch.utils.argparse import *

from dms_plots import dms_scatter_plot, dms_roc_plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_filepath', type=str, default='runs/cgnet_full_full_lmax=5__v0/2LZM-final_model.npz')
    parser.add_argument('--dms_input_filepath', type=optional_str, default='../T4_mutant_ddG.csv', # '../T4_mutant_ddG.csv', '/gscratch/spe/gvisan01/dms_data/AMIE_PSEAE_Whitehead/output_AMIE_PSEAE_Whitehead.csv'
                        help='Path to csv file containing DMS experiment data. If None, script returns all possible mutations in file named "all_muts.csv".')
    parser.add_argument('--dms_output_filepath', type=optional_str, default='runs/cgnet_full_full_lmax=5__v0/2LZM-final_model.csv')
    
    parser.add_argument('--dms_column', type=optional_str, default='ddG') # ddG
    parser.add_argument('--dms_label', type=optional_str, default=None) #r'stability effect, $\Delta\Delta G$')
    parser.add_argument('--dms_filepath', type=optional_str, default='runs/cgnet_full_full_lmax=5__v0/dGG-final_model.png') #r'stability effect, $\Delta\Delta G$')

    parser.add_argument('--binary_dms_column', type=optional_str, default=None) #'effect')
    parser.add_argument('--binary_dms_pos_value', type=optional_str, default=None) #'Destabilizing')
    parser.add_argument('--binary_dms_label', type=optional_str, default=None) #'Stability Effect\n(Destabilizing = 1, Neutral = 0)')
    parser.add_argument('--binary_dms_filepath', type=optional_str, default=None) # runs/cgnet_full_full_lmax=5__v0/dGG-final_model.png
    
    args = parser.parse_args()

    # load csv file with mutants and mutant values, append pseudoenergies to them, save the updated dataframe to the same filepath
    arrays = np.load(args.predictions_filepath)

    data_ids = arrays['data_ids']
    positions = np.array([int(data_id.split('_')[3]) for data_id in data_ids])

    positions = np.arange(1, positions.shape[0]+1)

    logits = arrays['logits']
    targets = arrays['targets']
    aa_targets = np.array([aa_to_one_letter[ind_to_aa[target]] for target in targets])
    guesses = np.argmax(logits, axis=-1)
    print('Accuracy: %.3f' % (np.count_nonzero(guesses == targets)/len(guesses)))

    # if dms file exists, check that position and wt at position math, within the DMS expeirment. Otherwise throw an error.
    if args.dms_input_filepath is not None:
        df = pd.read_csv(args.dms_input_filepath)

        for mutant in df['mutant']:
            if ':' in mutant:
                raise NotImplementedError('Multiple mutations prediction not yet implemented')
        
        dms_pos_N = np.array([int(mutant[1:-1]) for mutant in df['mutant']])
        unique_dms_pos = np.array(list(set(list(dms_pos_N))))
        dms_pos_to_new_pos = {}
        for i, pos in enumerate(unique_dms_pos):
            dms_pos_to_new_pos[pos] = i + 1
        dms_pos_N = np.array([dms_pos_to_new_pos[pos] for pos in dms_pos_N])

        dms_wt_N = np.array([mutant[0] for mutant in df['mutant']])
        dms_mut_N = np.array([mutant[-1] for mutant in df['mutant']])

        print(positions)
        print(np.array(list(set(list(dms_pos_N)))))

        # check DMS csv file and predictions are aligned
        for dms_pos, dms_wt in zip(dms_pos_N, dms_wt_N):
            for pos, wt in zip(positions, aa_targets):
                if dms_pos == pos:
                    if dms_wt == wt:
                        break
                    else:
                        raise Exception('DMS positions and PDB positions do not match.')
        
        pe_wt = [logits[np.where(positions == dms_pos)[0][0]][aa_to_ind[one_letter_to_aa[dms_wt]]] if np.where(positions == dms_pos)[0].shape[0] > 0 else np.nan for dms_pos, dms_wt in zip(dms_pos_N, dms_wt_N)]
        pe_mut = [logits[np.where(positions == dms_pos)[0][0]][aa_to_ind[one_letter_to_aa[dms_mut]]] if np.where(positions == dms_pos)[0].shape[0] > 0 else np.nan for dms_pos, dms_mut in zip(dms_pos_N, dms_mut_N)]
        df['pe_wt'] = pe_wt
        df['pe_mut'] = pe_mut

        log_proba_wt = [np.log(softmax(logits, axis=-1)[np.where(positions == dms_pos)[0][0]][aa_to_ind[one_letter_to_aa[dms_wt]]]) if np.where(positions == dms_pos)[0].shape[0] > 0 else np.nan for dms_pos, dms_wt in zip(dms_pos_N, dms_wt_N)]
        log_proba_mut = [np.log(softmax(logits, axis=-1)[np.where(positions == dms_pos)[0][0]][aa_to_ind[one_letter_to_aa[dms_mut]]]) if np.where(positions == dms_pos)[0].shape[0] > 0 else np.nan for dms_pos, dms_mut in zip(dms_pos_N, dms_mut_N)]
        df['log_proba_wt'] = log_proba_wt
        df['log_proba_mut'] = log_proba_mut

        df.to_csv(args.dms_output_filepath)
    else:
        raise NotImplementedError('Option with no DMS experiment provided not yet implemented.')

    # correlations
    df['log_proba_mut__minus__log_proba_wt'] = df['log_proba_mut'] - df['log_proba_wt']

    dms_comparison_json = {}

    if args.dms_column is not None:
        (pearson_r, pearson_pval), (spearman_r, spearman_pval) = dms_scatter_plot(df,
                                                                                  args.dms_column, 'log_proba_mut__minus__log_proba_wt',
                                                                                  dms_label=args.dms_label, pred_label=r'H-CNN Prediction',
                                                                                  filename = args.dms_filepath)
        dms_comparison_json['Pearson R'] = pearson_r
        dms_comparison_json['Pearson R - pval'] = pearson_pval
        dms_comparison_json['Spearman R'] = spearman_r
        dms_comparison_json['Spearman R - pval'] = spearman_pval
                                                                        

    # roc_curves for binary outputs
    if args.binary_dms_column is not None:
        roc_auc = dms_roc_plot(df,
                               args.binary_dms_column,
                               'log_proba_mut__minus__log_proba_wt',
                               dms_pos_value = args.binary_dms_pos_value,
                               dms_label = args.binary_dms_label,
                               filename = args.binary_dms_filepath)
        dms_comparison_json['ROC AUC'] = roc_auc
    
    if len(dms_comparison_json) != 0:
        dms_output_filepath_json = args.dms_output_filepath.replace('.csv', '.json')
        with open(dms_output_filepath_json, 'w+') as f:
            json.dump(dms_comparison_json, f, indent=2)
    