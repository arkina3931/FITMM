# coding: utf-8

"""
Main entry
# UPDATED
##########################
"""

import os
import argparse
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument(
        '--gpu_id',
        '--gpuid',
        type=int,
        default=0,
        help='physical GPU id to use when use_gpu=True'
    )
    parser.add_argument(
        '--ablation_variant',
        type=str,
        default='full',
        help='FITMM ablation variant, e.g. full / wo_freq / wo_item_graph'
    )

    config_dict = {
        'gpu_id': None,
        'ablation_variant': 'full',
    }

    args, _ = parser.parse_known_args()
    config_dict['gpu_id'] = args.gpu_id
    config_dict['ablation_variant'] = args.ablation_variant

    from utils.quick_start import quick_start

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
