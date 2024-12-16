import argparse
import os

import numpy as np
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from copy import deepcopy
from rdkit import Chem

from utils import misc
from utils.evaluation import scoring_func
from utils.evaluation.docking_vina import VinaDockingTask
from multiprocessing import Pool
from functools import partial
from glob import glob
import pickle
import re
from datasets.protein_ligand import parse_sdf_file_mol



def eval_single_datapoint(ligand_rdmol_list, protein_path, args, center):

    results = []
    
    n_eval_success = 0

    for rdmol in tqdm(ligand_rdmol_list):
        
        if rdmol is None:
            results.append({
                'mol': None,
                'smiles': None,
                'protein_path': protein_path,
            })
            continue

        try:
            Chem.SanitizeMol(rdmol)
        except Chem.rdchem.AtomValenceException as e:
            err = e
            N4_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
            index = N4_valence.findall(err.args[0])
            if len(index) > 0:
                rdmol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
                Chem.SanitizeMol(rdmol)
                
        smiles = Chem.MolToSmiles(rdmol)
        
        print(smiles)

        if '.' in smiles:
            results.append({
                'mol': rdmol,
                'smiles': smiles,
                'protein_path': protein_path,
            })
            continue
        
        mol = rdmol

        chem_results = scoring_func.get_chem(mol)

        vina_task = VinaDockingTask(
            protein_path=protein_path,
            ligand_rdmol=deepcopy(mol),
            size_factor=None,
            center=center.tolist(),
        )
            
        score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
        minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
        vina_results = {
            'score_only': score_only_results,
            'minimize': minimize_results
        }
        if args.docking_mode == 'vina_full':
            dock_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
            vina_results.update({
                'dock': dock_results,
            })
        elif args.docking_mode == 'vina_score':
            pass
        else:
            raise NotImplementedError
        
        n_eval_success += 1
            
        results.append({
            'mol': mol,
            'smiles': smiles,
            'protein_path': protein_path,
            'chem_results': chem_results,
            'vina': vina_results,
        })
    logger.info(f'Evaluate No {id} done! {len(ligand_rdmol_list)} samples in total. {n_eval_success} eval success!')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synergy_idx', type=int)  
    parser.add_argument('--sample_path', type=str)
    
    parser.add_argument('-n', '--eval_num_examples', type=int, default=10)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--docking_mode', type=str, default='vina_full',
                        choices=['vina_full', 'vina_score'])
    parser.add_argument('--exhaustiveness', type=int, default=32)
    parser.add_argument('--result_path', type=str, required=True)
    
    args = parser.parse_args()
    
    with open('data/processed/drug_synergy/synergy_idx_list.pkl', 'rb') as f:
        synergy_idx_list = pickle.load(f)
    with open('data/processed/dock/index_dict.pkl', 'rb') as f:
        index_dict = pickle.load(f)
    idx_to_smiles = index_dict['idx_to_smiles']

    idx1, idx2 = synergy_idx_list[args.synergy_idx]
    smiles1 = idx_to_smiles[idx1]
    smiles2 = idx_to_smiles[idx2]
    
    
    if os.path.exists(os.path.join(args.result_path, f'{idx1}/{idx2}/eval_all.pkl')):
        exit()

    if args.result_path:
        os.makedirs(os.path.join(args.result_path, f'{idx1}/{idx2}'), exist_ok=True)
    logger = misc.get_logger('evaluate', args.result_path)
    logger.info(args)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    logger.info(f'synergy_idx: ({idx1}, {idx2})')

    
    ligand_meta_file = os.path.join(args.sample_path, f'{idx1}/{idx2}/sample.pt')
    assert os.path.exists(ligand_meta_file), f"{ligand_meta_file}"
    ligand_meta = torch.load(ligand_meta_file)
    ligand_rdmol_list = ligand_meta['mols'][:args.eval_num_examples]
    
    
    protein_path = glob(f"data/processed/dock/ligand_protein_dataset_v2/{smiles2}/*/protein_clean.pdb")[0]
    anchor_ligand_dict = parse_sdf_file_mol(glob(f"data/processed/dock/ligand_protein_dataset_v2/{smiles2}/*/ligand.sdf")[0])

    
    testset_results = eval_single_datapoint(ligand_rdmol_list, protein_path, args, center=anchor_ligand_dict['center_of_mass'])

    if args.result_path:
        with open(os.path.join(args.result_path, f'{idx1}/{idx2}/eval_all.pkl'), 'wb') as f:
            pickle.dump(testset_results, f)
    
    print("Finished")
    

    qed = [x['chem_results']['qed'] for x in testset_results if x['mol'] is not None]
    sa = [x['chem_results']['sa'] for x in testset_results if x['mol'] is not None]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    if args.docking_mode in ['vina', 'qvina']:
        vina = [x['vina'][0]['affinity'] for x in testset_results if x['mol'] is not None]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_full', 'vina_score']:
        vina_score_only = [x['vina']['score_only'][0]['affinity'] for x in testset_results if x['mol'] is not None]
        vina_min = [x['vina']['minimize'][0]['affinity'] for x in testset_results if x['mol'] is not None]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_full':
            vina_dock = [x['vina']['dock'][0]['affinity'] for x in testset_results if x['mol'] is not None]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))
