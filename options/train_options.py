from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
        parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
        parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
        parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")

        parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
        parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")

        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        self.isTrain = True
        return parser