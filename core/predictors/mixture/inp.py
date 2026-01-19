import torch
from tap import Tap
import os


class CommonArgs(Tap):
    optimization: bool = False
    working_dir = os.getcwd()
    base_dir, _ = os.path.split(working_dir)
    dir: str = base_dir + '/'
    split_ratio: tuple = (0.8, 0.1, 0.1)
    seed: int = 0
    model_path: str = os.path.join(working_dir, 'trained_models', 'GHsolvQM')
    output_dir: str = dir + "ModelPredictions"
    make_plots: bool = True
    scale: str = "standard"  # standard or minmax
    scale_features: bool = False
    use_same_scaler_for_features: bool = False
    # random or scaffold or wo_solvents (the latter is based on random split) or kmeans or onecross
    split: str = "random"
    kmeans_split_base: str = "solvent"  # solvent or solute depending on if you want first or second molecule
    save_memory: bool = False

    # for featurization
    property: str = "solvation"  # alternatives are solvation, Tm and logS
    add_hydrogens_to_solvent: bool = False  # adds hydrogens to solvents (first column) if you have 2 input smiles
    mix: bool = False  # features are fractions of the different molecules in the same order
    ####################################################################################################################
    # for active learning
    uncertainty: bool = False  # calculate and output aleotoric uncertainties
    ensemble_variance: bool = False  # calculate and output ensemble variance, epi
    # number or adaptpercent-n for n% of training set
    active_learning_batch_size: str = "adaptpercent-10"
    active_learning_iterations: int = 100
    # how to select data, options are: epistemic, total and random, epi_mol, epi_scaled
    # (for epi unc on scaled predictions)
    data_selection: str = "epistemic"
    restart_al: bool = False
    active_learning_split_ratio = (
        0.3,
        0.4,
        0.3,
    )  # split between initial train data, experimental data and test set
    # for training
    epochs: int = 10
    batch_size: int = 50
    loss_metric: str = "mse"

    # mpn or ffn or none or onlylast or mpn1 or onlylast1 if you have only one molecule
    learning_rates: tuple = (0.001, 0.0001, 0.001)  # initial, final, max
    warm_up_epochs: float = (
        2.0  # you need min 1 with adam optimizer and Noam learning rate scheduler
    )
    lr_scheduler: str = "Noam"  # Noam or Step or Exponential
    # in case of step
    step_size: int = 10
    step_decay: float = 0.2
    # in case of exponential
    exponential_decay: float = 0.1
    minimize_score: bool = True

    cuda: bool = False and (torch.backends.mps.is_available() or torch.cuda.is_available())
    if cuda:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda')
    else:
        device = torch.device('cpu')
    gpu: int = 4
    # results
    print_weigths: bool = False
    postprocess: bool = False

    # for mpn
    depth: int = 4
    mpn_hidden: int = 200
    mpn_dropout: float = 0.00
    mpn_activation: str = "LeakyReLU"
    mpn_bias: bool = False
    morgan_fingerprint: str = (
        "None"  # None, only_solvent or All #if you want morgan fingerprints
    )
    morgan_bits: int = 16
    morgan_radius: int = 2
    aggregation: str = "mean"
    # make sure your solvent is the first in the input file
    # self.dummy_atom_for_single_atoms = True

    # for attention
    attention: bool = False  # True or false
    att_hidden: int = 200
    att_dropout: float = 0.0
    att_bias: bool = False
    att_activation: str = "ReLU"
    att_normalize: str = "sigmoid"  # sigmoid or softmax or logsigmoid of logsoftmax or None
    att_first_normalize: bool = False

    # for ffn
    ffn_hidden: int = 500
    ffn_num_layers: int = 4
    ffn_dropout: float = 0.00
    ffn_activation: str = "LeakyReLU"
    ffn_bias: bool = True


class TrainArgs(CommonArgs):
    # if the entire dataset is infinite dilute set this to True
    solute = False
    input_file: str = CommonArgs.dir + "Data/UGent/Viscosity_training_ln.csv"
    num_folds: int = 10
    max_num_mols: int = 4
    num_models: int = 10
    num_targets: int = 1
    f_mol_size: int = 2
    num_features: int = 0
    max_molecules: int = -1  # -1 for all
    pretraining_fix: str = "none"
    pretraining: bool = False
    if pretraining:
        pretraining_path: list = [
            CommonArgs.dir + f"solvation_predictor/trained_models/Viscosity/model{i}.pt" for i in range(0, 10)
        ]
    # The headers of the input file
    solute_headers: list = []
    solvent_headers: list = ["fuel1_inchi", "fuel2_inchi", "fuel3_inchi", "fuel4_inchi"]
    target_headers: list = ["Viscosity"]
    features_headers: list = []
    molefrac_headers: list = ["frac_fuel1 (molar)", "frac_fuel2 (molar)", "frac_fuel3 (molar)"]
    delimiter: str = ","


class PredictArgs(CommonArgs):
    input_file: str = os.path.join(CommonArgs.base_dir, 'Data', 'UGent/Viscosity_test_ln.csv')
    model_path_root: str = os.path.join(CommonArgs.working_dir, 'trained_models', 'Viscosity')
    # FIXED: Don't list directory at class definition time - do it lazily
    model_path = None  # Will be set when needed
    output_dir = os.path.join(CommonArgs.base_dir, "MyOutput")
    get_molecular_embedding = "solvent"  # save the embedding for solvents

    # if the entire dataset is infinite dilute set this to True
    solute = False
    max_num_mols: int = 4
    num_targets: int = 1
    f_mol_size: int = 2
    num_features: int = 0
    max_molecules: int = -1  # -1 for all
    # The headers of the input file
    solute_headers: list = []
    solvent_headers: list = ["fuel1_inchi", "fuel2_inchi", "fuel3_inchi", "fuel4_inchi"]
    target_headers: list = ["Viscosity"]
    features_headers: list = []
    molefrac_headers: list = ["frac_fuel1 (molar)", "frac_fuel2 (molar)", "frac_fuel3 (molar)"]
    delimiter: str = ","
    
    def __post_init__(self):
        """Set model_path lazily after initialization"""
        if self.model_path is None and os.path.exists(self.model_path_root):
            self.model_path = [f for f in os.listdir(self.model_path_root) if '.pt' in f]