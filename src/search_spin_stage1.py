# coding: utf-8
import argparse
import os


def parse_int_list(raw):
    return [int(token.strip()) for token in raw.split(',') if token.strip()]


def parse_float_list(raw):
    return [float(token.strip()) for token in raw.split(',') if token.strip()]


def parse_str_list(raw):
    return [token.strip() for token in raw.split(',') if token.strip()]


def parse_bool_token(raw):
    token = str(raw).strip().lower()
    if token in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if token in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise ValueError(f'Cannot parse boolean value from: {raw}')


def parse_bool_list(raw):
    return [parse_bool_token(token) for token in raw.split(',') if token.strip()]


def normalize_search_value(values):
    if len(values) == 1:
        return values[0]
    return values


def count_combinations(search_space, hyper_parameters):
    total = 1
    for key in hyper_parameters:
        value = search_space[key]
        if isinstance(value, list):
            total *= len(value)
    return total


def build_parser():
    parser = argparse.ArgumentParser(description='Hyper-parameter search for SPIN Stage 1.')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='Dataset name.')
    parser.add_argument('--gpu_id', '--gpuid', type=int, default=0, help='Physical GPU id.')
    parser.add_argument('--save_model', type=parse_bool_token, default=True, help='Whether to save checkpoints.')
    parser.add_argument('--dry_run', action='store_true', help='Print the resolved search space and exit.')

    parser.add_argument('--seed', type=parse_int_list, default=[999], help='Comma-separated seeds.')
    parser.add_argument('--embedding_size', type=parse_int_list, default=[64], help='Embedding size grid.')
    parser.add_argument('--knn_k', type=parse_int_list, default=[10], help='KNN graph top-k grid.')
    parser.add_argument('--aggr_mode', type=parse_str_list, default=['add'], help='GCN aggregation mode grid.')
    parser.add_argument('--reg_weight', type=parse_float_list, default=[0.001], help='Regularization weight grid.')
    parser.add_argument('--learning_rate', type=parse_float_list, default=[1e-4], help='Learning rate grid.')
    parser.add_argument('--ib_weight', type=parse_float_list, default=[1.0], help='IB loss weight grid.')
    parser.add_argument('--num_layers', type=parse_int_list, default=[2], help='User-item GCN layer grid.')
    parser.add_argument('--num_freq_bands', type=parse_int_list, default=[3], help='Frequency band grid.')
    parser.add_argument('--ib_direction', type=parse_str_list, default=['Pos'], help='IB direction grid.')
    parser.add_argument('--ib_alpha', type=parse_float_list, default=[1.0], help='IB alpha grid.')
    parser.add_argument('--ib_mu', type=parse_float_list, default=[1.0], help='IB mu grid.')
    parser.add_argument('--ib_phi_plus', type=parse_float_list, default=[0.0], help='IB phi_plus grid.')

    parser.add_argument(
        '--spin_dual_weight',
        type=parse_float_list,
        default=[0.02, 0.05, 0.1],
        help='Residual dual-stream weight grid.',
    )
    parser.add_argument(
        '--spin_router_hidden_dim',
        type=parse_int_list,
        default=[64, 128],
        help='Router hidden dimension grid.',
    )
    parser.add_argument(
        '--spin_router_dropout',
        type=parse_float_list,
        default=[0.0, 0.1],
        help='Router dropout grid.',
    )
    parser.add_argument(
        '--spin_use_activity_popularity',
        type=parse_bool_list,
        default=[True],
        help='Whether to use degree-derived prior.',
    )
    parser.add_argument(
        '--spin_score_mode',
        type=parse_str_list,
        default=['residual'],
        help='Score mode grid. Recommended default: residual.',
    )
    parser.add_argument(
        '--spin_pop_alpha',
        type=parse_float_list,
        default=[1.0],
        help='Popular stream weight for two_stream scoring.',
    )
    parser.add_argument(
        '--spin_niche_beta',
        type=parse_float_list,
        default=[1.0],
        help='Niche stream weight for two_stream scoring.',
    )
    parser.add_argument(
        '--spin_orth_weight',
        type=parse_float_list,
        default=[0.0],
        help='Orthogonality loss weight grid.',
    )
    parser.add_argument(
        '--spin_enable_dual_stream',
        type=parse_bool_list,
        default=[True],
        help='Whether to enable the dual-stream residual branch.',
    )

    return parser


def build_config_dict(args):
    hyper_parameters = [
        'seed',
        'embedding_size',
        'knn_k',
        'aggr_mode',
        'reg_weight',
        'learning_rate',
        'ib_weight',
        'num_layers',
        'num_freq_bands',
        'ib_direction',
        'ib_alpha',
        'ib_mu',
        'ib_phi_plus',
        'spin_dual_weight',
        'spin_router_hidden_dim',
        'spin_router_dropout',
        'spin_use_activity_popularity',
        'spin_score_mode',
        'spin_pop_alpha',
        'spin_niche_beta',
        'spin_orth_weight',
        'spin_enable_dual_stream',
    ]

    search_space = {
        'gpu_id': args.gpu_id,
        'ablation_variant': 'full',
        'seed': args.seed,
        'embedding_size': args.embedding_size,
        'knn_k': args.knn_k,
        'aggr_mode': args.aggr_mode,
        'reg_weight': args.reg_weight,
        'learning_rate': args.learning_rate,
        'ib_weight': args.ib_weight,
        'num_layers': args.num_layers,
        'num_freq_bands': args.num_freq_bands,
        'ib_direction': args.ib_direction,
        'ib_alpha': args.ib_alpha,
        'ib_mu': args.ib_mu,
        'ib_phi_plus': args.ib_phi_plus,
        'spin_dual_weight': args.spin_dual_weight,
        'spin_router_hidden_dim': args.spin_router_hidden_dim,
        'spin_router_dropout': args.spin_router_dropout,
        'spin_use_activity_popularity': args.spin_use_activity_popularity,
        'spin_score_mode': args.spin_score_mode,
        'spin_pop_alpha': args.spin_pop_alpha,
        'spin_niche_beta': args.spin_niche_beta,
        'spin_orth_weight': args.spin_orth_weight,
        'spin_enable_dual_stream': args.spin_enable_dual_stream,
        'hyper_parameters': hyper_parameters,
    }

    config_dict = {}
    for key, value in search_space.items():
        if key == 'hyper_parameters':
            config_dict[key] = value
        elif key in hyper_parameters:
            config_dict[key] = value
        else:
            config_dict[key] = normalize_search_value(value) if isinstance(value, list) else value

    return config_dict, search_space, hyper_parameters


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = build_parser()
    args = parser.parse_args()

    config_dict, search_space, hyper_parameters = build_config_dict(args)
    total = count_combinations(search_space, hyper_parameters)

    print('Resolved SPIN Stage 1 search space:')
    for key in hyper_parameters:
        print(f'  {key}: {search_space[key]}')
    print(f'Total combinations: {total}')

    if args.dry_run:
        return

    from utils.quick_start import quick_start

    quick_start(
        model='SPIN',
        dataset=args.dataset,
        config_dict=config_dict,
        save_model=args.save_model,
    )


if __name__ == '__main__':
    main()
