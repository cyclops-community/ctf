def add_arguments(parser):
    parser.add_argument(
        '--num_iter',
        type=int,
        default=10,
        metavar='int',
        help='Number of iterations (default: 10)')
    parser.add_argument(
        '--s_start',
        type=int,
        default=100,
        metavar='int',
        help='Tensor starting dimension (default: 100)')
    parser.add_argument(
        '--s_end',
        type=int,
        default=400,
        metavar='int',
        help='Tensor max dimension (default: 400)')
    parser.add_argument(
        '--mult',
        type=float,
        default=2,
        metavar='float',
        help='Multiplier by which to grow dimension (default: 2)')
    parser.add_argument(
        '--R',
        type=int,
        default=40,
        metavar='int',
        help='Second dimension of matrix (default: 40)')
    parser.add_argument(
        '--sp',
        type=int,
        default=1,
        metavar='int',
        help='Whether to use sparse format (default: 1)')
    parser.add_argument(
        '--sp_init',
        type=float,
        default=1.,
        metavar='float',
        help='Initial sparsity fraction (default: 1)')
    parser.add_argument(
        '--use_cust_MTTKRP',
        type=int,
        default=1,
        metavar='int',
        help='Whether to use MTTKRP routine (default: 1/True)')
