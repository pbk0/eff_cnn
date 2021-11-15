import sys
import zipfile
import itertools
import pandas as pd
import tqdm
import sys
import pathlib
_home = pathlib.Path.home()
sys.path.extend(
    [
        (_home / "scandal").as_posix(),
        (_home / "toolcraft").as_posix(),
        (_home / "Documents" / "Github" / "toolcraft").as_posix(),
        (_home / "Documents" / "Github" / "RU").as_posix(),
        "..",
    ]
)
import typing as t
import pathlib
import numpy as np
import enum
from sklearn.model_selection import train_test_split
import tqdm
import plotly.graph_objects as go
import shutil
import os
import pickle
import plotly.express as px
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from src.clr import OneCycleLR
from src import dataLoaders
from src import preproces
from src import models
# from src import mt

IS_ON_NXDI = (pathlib.Path.home() / "__is_on_nxdi__").exists()
ROOT_DIR = pathlib.Path(__file__).parent
if IS_ON_NXDI:
    RESULTS_DIR = pathlib.Path(os.environ["VAPATH"] + "/eff_cnn")
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)
else:
    RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"
REPORTS_DIR = ROOT_DIR / "reports"
NUM_ATTACKS_PER_EXPERIMENT = 20
NUM_EXPERIMENTS = 50


def fig_update_layout(_fig):
    # xaxes and yaxes
    # https://plotly.com/python/axes/
    # _fig.update_xaxes(title_font=dict(size=18, family='Courier', color='crimson'))
    # _fig.update_yaxes(title_font=dict(size=18, family='Courier', color='crimson'))

    # template
    # https://plotly.com/python/templates/
    # _template = go.layout.Template()
    _template = dict(
        layout=go.Layout(
            title_font=dict(family="Rockwell", size=24),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
        ),
    )
    _template = 'ggplot2'
    _fig.update_layout(template=_template)


def preprocess_predictions(predictions, all_guess_targets, num_examples,
                           num_guesses) -> np.ndarray:
    # make copy
    predictions = predictions.copy()

    # Add small positive value
    # note that we set any o or negative probability to smallest
    # possible positive number so that np.log does not
    # result to -np.inf
    predictions[predictions <= 1e-45] = 1e-45

    # Sort based on guessed targets
    sorted_predictions = predictions[
        np.asarray(
            [np.arange(num_examples)]
        ).repeat(num_guesses, axis=0).T,
        all_guess_targets
    ]

    # take negative logs
    sorted_neg_log_preds = -np.log(sorted_predictions)

    # return
    return sorted_neg_log_preds


def compute_ranks(predictions, all_guess_targets, correct_key,
                  num_attacks) -> np.ndarray:
    # num_examples and num_guesses
    num_examples = predictions.shape[0]
    num_guesses = 256

    # some buffers
    all_ranks = np.zeros((num_attacks, num_examples), np.uint8)

    # fix seed for deterministic behaviour
    np.random.seed(123456)

    # get sorted_neg_log_preds
    sorted_neg_log_preds = preprocess_predictions(predictions, all_guess_targets,
                                                  num_examples, num_guesses)

    # loop over
    for attack_id in tqdm.trange(num_attacks):
        # first shuffle for simulating random experiment
        np.random.shuffle(sorted_neg_log_preds)

        # cum sum
        sorted_neg_log_preds_cum_sum = np.cumsum(sorted_neg_log_preds, axis=0)

        # compute rank
        ranks_for_all_guesses = sorted_neg_log_preds_cum_sum.argsort().argsort()

        # set correct rank
        all_ranks[attack_id, :] = ranks_for_all_guesses[:, correct_key]

    # return
    return all_ranks


class Dataset(enum.Enum):
    ascad_v1_fk_0 = enum.auto()
    ascad_v1_fk_0_noisy = enum.auto()
    ascad_v1_vk_0 = enum.auto()
    ascad_v1_vk_0_noisy = enum.auto()
    ascad_v1_fk_50 = enum.auto()
    ascad_v1_fk_100 = enum.auto()
    aes_hd = enum.auto()
    aes_rd = enum.auto()
    dpav4 = enum.auto()

    @property
    def rank_plot_until(self) -> int:
        if self in [
            self.ascad_v1_fk_0, self.ascad_v1_fk_50, self.ascad_v1_fk_100,
        ]:
            return 250
        elif self in [self.ascad_v1_vk_0, ]:
            return 1000
        elif self in [self.ascad_v1_fk_0_noisy, self.ascad_v1_vk_0_noisy]:
            return 3000
        elif self is self.aes_hd:
            return 1200
        elif self is self.aes_rd:
            return 10
        elif self is self.dpav4:
            return 10
        else:
            raise Exception(f"Unsupported dataset `{self.name}`")

    def traces_to_use_for_rank(self) -> t.List[int]:
        if self is self.ascad_v1_fk_0:
            return [25, 50, 100, 300, 400, 600]
        elif self is self.ascad_v1_vk_0:
            return [100, 150, 200, 250, 300, 350]
        elif self is self.ascad_v1_fk_0_noisy:
            return [50, 100, 250, 700, 1000, 1400]
        elif self is self.ascad_v1_vk_0_noisy:
            return [200, 300, 400, 500, 600, 750]
        elif self is self.ascad_v1_fk_50:
            return [50, 100, 170, 400, 600, 800]
        elif self is self.ascad_v1_fk_100:
            return [50, 100, 170, 400, 600, 800]
        else:
            raise Exception(
                f"Dataset {self} is not supported"
            )

    def get(
            self
    ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # Load a dataset (see src/dataLoaders.py)
        if self is self.ascad_v1_fk_0:
            _data = dataLoaders.load_ascad(f'./../datasets/ASCAD_dataset/ASCAD.h5')
        elif self is self.ascad_v1_fk_0_noisy:
            _data = dataLoaders.load_ascad(f'./../datasets/ASCAD_dataset/ASCAD.h5',
                                           add_noise=2.0)
        elif self is self.ascad_v1_vk_0:
            _data = dataLoaders.load_ascad(f'./../datasets/ascad-variable.h5')
        elif self is self.ascad_v1_vk_0_noisy:
            _data = dataLoaders.load_ascad(f'./../datasets/ascad-variable.h5',
                                           add_noise=2.0)
        elif self is self.ascad_v1_fk_50:
            _data = dataLoaders.load_ascad(
                f'./../datasets/ASCAD_dataset/ASCAD_desync50.h5')
        elif self is self.ascad_v1_fk_100:
            _data = dataLoaders.load_ascad(
                f'./../datasets/ASCAD_dataset/ASCAD_desync100.h5')
        elif self is self.aes_hd:
            _data = dataLoaders.load_aes_hd(f'./../datasets/AES_HD_dataset/')
        elif self is self.aes_rd:
            _data = dataLoaders.load_aes_rd(f'./../datasets/AES_RD_dataset/')
        elif self is self.dpav4:
            _data = dataLoaders.load_dpav4(f'./../datasets/DPAv4_dataset/')
        else:
            raise Exception(f"Dataset `{self}` is not supported")

        return _data


class Preprocessor(enum.Enum):
    none = enum.auto()
    feature_standardization = enum.auto()
    horizontal_standardization = enum.auto()

    @property
    def preprocess_fn(self) -> t.Callable:

        # get preprocessor
        if self is self.none:
            _preprocessor = preproces.no_preprocessing
        elif self is self.feature_standardization:
            _preprocessor = preproces.feature_standardization
        elif self is self.horizontal_standardization:
            _preprocessor = preproces.horizontal_standardization
        else:
            raise Exception(f"Preprocessor `{self}` is not supported")

        return _preprocessor

    def apply(
            self, dataset: Dataset, hw_leakage_model: bool = False
    ) -> t.Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # get dataset
        X_profiling, Y_profiling, X_attack, targets, key_attack = dataset.get()

        # Preprocess the data (see src/preproces.py)
        X_profiling_processed, X_attack_processed = self.preprocess_fn(X_profiling,
                                                                       X_attack)

        # compute hamming weight if requested
        if hw_leakage_model:
            Y_profiling = np.vectorize(
                lambda x: bin(x).count("1")
            )(Y_profiling).astype(np.uint8)
            targets = np.vectorize(
                lambda x: bin(x).count("1")
            )(targets).astype(np.uint8)

        # Automatic split (90/10)
        # Random_state is set to get reproducible splits
        # model weights are initialised randomly (can also saved in case we encouter weird things)
        tracesTrain, tracesVal, labelsTrain, labelsVal = train_test_split(
            X_profiling_processed, Y_profiling, test_size=0.1, random_state=0)

        # return
        return tracesTrain, tracesVal, labelsTrain, labelsVal, X_attack_processed, targets, key_attack


class Model(enum.Enum):
    ascad_mlp = enum.auto()
    ascad_cnn = enum.auto()
    ascad_cnn2 = enum.auto()
    eff_cnn = enum.auto()

    s_eff_cnn_id = enum.auto()
    s_eff_cnn_hw = enum.auto()
    aisy_mlp_id = enum.auto()
    aisy_mlp_hw = enum.auto()

    @property
    def is_hw_leakage_model(self) -> bool:
        if self in [self.aisy_mlp_hw, self.s_eff_cnn_hw]:
            return True
        return False

    @property
    def is_mcovc(self) -> bool:
        if self.name.startswith("mcovc"):
            return True
        else:
            return False

    # noinspection DuplicatedCode
    def make_fn(self, dataset: Dataset) -> t.Callable:
        if self.is_mcovc:
            raise Exception(
                "mcovc models are not supported"
            )
        if self in [self.ascad_mlp, ]:
            if dataset in [Dataset.ascad_v1_fk_0, Dataset.ascad_v1_fk_50,
                           Dataset.ascad_v1_fk_100, ]:
                return models.ascad_mlp_best
        elif self in [self.ascad_cnn, ]:
            if dataset in [Dataset.ascad_v1_fk_0, Dataset.ascad_v1_fk_50,
                           Dataset.ascad_v1_fk_100, ]:
                return models.ascad_cnn_best
        elif self in [self.ascad_cnn2, ]:
            if dataset in [Dataset.ascad_v1_vk_0, ]:
                return models.ascad_cnn_best2
        elif self is self.eff_cnn:
            if dataset is Dataset.ascad_v1_fk_0:
                return models.zaid_ascad_desync_0
            if dataset is Dataset.ascad_v1_fk_50:
                return models.zaid_ascad_desync_50
            if dataset is Dataset.ascad_v1_fk_100:
                return models.zaid_ascad_desync_100
            if dataset is Dataset.aes_hd:
                return models.zaid_aes_hd
            if dataset is Dataset.aes_rd:
                return models.zaid_aes_rd
            if dataset is Dataset.dpav4:
                return models.zaid_dpav4
        elif self is self.s_eff_cnn_id:
            if dataset in [Dataset.ascad_v1_fk_0, Dataset.ascad_v1_fk_0_noisy, ]:
                return models.noConv1_ascad_desync_0
            if dataset is Dataset.ascad_v1_fk_50:
                return models.noConv1_ascad_desync_50
            if dataset is Dataset.ascad_v1_fk_100:
                return models.noConv1_ascad_desync_100
            if dataset is Dataset.aes_hd:
                return models.noConv1_aes_hd
            if dataset is Dataset.aes_rd:
                return models.noConv1_aes_rd
            if dataset is Dataset.dpav4:
                return models.noConv1_dpav4
        elif self is self.s_eff_cnn_hw:
            if dataset in [Dataset.ascad_v1_fk_0, Dataset.ascad_v1_fk_0_noisy, ]:
                return models.noConv1_ascad_desync_0_hw
            if dataset is Dataset.ascad_v1_fk_50:
                return models.noConv1_ascad_desync_50_hw
            if dataset is Dataset.ascad_v1_fk_100:
                return models.noConv1_ascad_desync_100_hw
        elif self is self.aisy_mlp_hw:
            if dataset in [Dataset.ascad_v1_fk_0, Dataset.ascad_v1_fk_0_noisy, ]:
                return models.aisy_ascad_f_hw_mlp
            if dataset in [Dataset.ascad_v1_vk_0, Dataset.ascad_v1_vk_0_noisy, ]:
                return models.aisy_ascad_r_hw_mlp
        elif self is self.aisy_mlp_id:
            if dataset in [Dataset.ascad_v1_fk_0, Dataset.ascad_v1_fk_0_noisy, ]:
                return models.aisy_ascad_f_id_mlp
            if dataset in [Dataset.ascad_v1_vk_0, Dataset.ascad_v1_vk_0_noisy, ]:
                return models.aisy_ascad_r_id_mlp
        else:
            raise Exception(f"Model `{self}` is not supported ...")
        raise Exception(
            f"Dataset `{dataset}` cannot be used with model `{self}` ..."
        )


class Params(t.NamedTuple):
    epochs: int
    batch_size: int
    learning_rate: float
    one_cycle_lr: bool
    preprocessor: Preprocessor


class ExperimentType(enum.Enum):
    original = enum.auto()
    early_stopping = enum.auto()
    over_fit = enum.auto()
    mcovc = enum.auto()


MODELS_TO_TRY = [
    Model.s_eff_cnn_id, Model.s_eff_cnn_hw,
    # Model.aisy_mlp_id, Model.aisy_mlp_hw,
    # Model.ascad_cnn2,
]
DATASETS_TO_TRY = [
    Dataset.ascad_v1_fk_0, Dataset.ascad_v1_vk_0,
    Dataset.ascad_v1_fk_0_noisy, Dataset.ascad_v1_vk_0_noisy,
    Dataset.ascad_v1_fk_50, Dataset.ascad_v1_fk_100,
]
EXPERIMENT_TYPES_TO_TRY = [
    ExperimentType.original,
    ExperimentType.early_stopping,
    # ExperimentType.over_fit,
]
DEFAULT_PARAMS = {
    Dataset.ascad_v1_fk_0: {
        Model.ascad_mlp: Params(
            epochs=200, batch_size=100, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.none,
        ),
        Model.ascad_cnn: Params(
            epochs=75, batch_size=200, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.none,
        ),
        Model.eff_cnn: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.s_eff_cnn_id: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.aisy_mlp_id: Params(
            epochs=50, batch_size=32, learning_rate=5e-4, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.s_eff_cnn_hw: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.aisy_mlp_hw: Params(
            epochs=50, batch_size=32, learning_rate=5e-4, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
    },
    Dataset.ascad_v1_fk_0_noisy: {
        Model.s_eff_cnn_id: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.aisy_mlp_id: Params(
            epochs=50, batch_size=32, learning_rate=5e-4, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.s_eff_cnn_hw: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.aisy_mlp_hw: Params(
            epochs=50, batch_size=32, learning_rate=5e-4, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
    },
    Dataset.ascad_v1_vk_0: {
        Model.ascad_cnn2: Params(
            epochs=75, batch_size=200, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.none,
        ),
        Model.aisy_mlp_hw: Params(
            epochs=50, batch_size=32, learning_rate=5e-4, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.aisy_mlp_id: Params(
            epochs=50, batch_size=32, learning_rate=5e-4, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
    },
    Dataset.ascad_v1_vk_0_noisy: {
        Model.aisy_mlp_hw: Params(
            epochs=50, batch_size=32, learning_rate=5e-4, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.aisy_mlp_id: Params(
            epochs=50, batch_size=32, learning_rate=5e-4, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
    },
    Dataset.ascad_v1_fk_50: {
        Model.ascad_mlp: Params(
            epochs=200, batch_size=100, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.none,
        ),
        Model.ascad_cnn: Params(
            epochs=75, batch_size=200, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.none,
        ),
        Model.eff_cnn: Params(
            epochs=50, batch_size=256, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
        Model.s_eff_cnn_id: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
        Model.s_eff_cnn_hw: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
    },
    Dataset.ascad_v1_fk_100: {
        Model.ascad_mlp: Params(
            epochs=200, batch_size=100, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.none,
        ),
        Model.ascad_cnn: Params(
            epochs=75, batch_size=200, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.none,
        ),
        Model.eff_cnn: Params(
            epochs=50, batch_size=256, learning_rate=1e-2, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
        Model.s_eff_cnn_id: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
        Model.s_eff_cnn_hw: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
    },
    Dataset.aes_hd: {
        Model.eff_cnn: Params(
            epochs=20, batch_size=256, learning_rate=1e-3, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.s_eff_cnn_id: Params(
            epochs=20, batch_size=256, learning_rate=1e-3, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
    },
    Dataset.aes_rd: {
        Model.eff_cnn: Params(
            epochs=50, batch_size=50, learning_rate=10e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
        Model.s_eff_cnn_id: Params(
            epochs=50, batch_size=50, learning_rate=10e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
    },
    Dataset.dpav4: {
        Model.eff_cnn: Params(
            epochs=50, batch_size=50, learning_rate=1e-3, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.s_eff_cnn_id: Params(
            epochs=50, batch_size=50, learning_rate=1e-3, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
    },
}  # type: t.Dict[Dataset, t.Dict[Model, Params]]


class Experiment(t.NamedTuple):
    id: int
    dataset: Dataset
    model: Model
    type: ExperimentType

    @property
    def name(self) -> str:
        return f"{self.type.name}-{self.dataset.name}-{self.model.name}-[{self.id}]"

    @property
    def store_dir(self) -> pathlib.Path:
        return RESULTS_DIR / self.type.name / self.dataset.name / self.model.name / str(
            self.id)

    @property
    def model_file_path(self) -> pathlib.Path:
        return self.store_dir / "model.hdf5"

    @property
    def history_file_path(self) -> pathlib.Path:
        return self.store_dir / "history.pickle"

    @property
    def ranks_file_path(self) -> pathlib.Path:
        return self.store_dir / "ranks.npy"

    @property
    def is_executing_file_path(self) -> pathlib.Path:
        return self.store_dir / "__is_executing__"

    @property
    def is_executing(self) -> bool:
        return self.is_executing_file_path.exists()

    @property
    def is_zipped(self) -> bool:
        _zip_file_path = ROOT_DIR / "results.zip"
        if _zip_file_path.exists():
            _zip_file = zipfile.ZipFile(_zip_file_path, 'r')
            _ranks_archive_name = "/".join(self.ranks_file_path.parts[-6:])
            if _ranks_archive_name in _zip_file.NameToInfo.keys():
                return True
            else:
                return False
        else:
            return False

    @property
    def is_done(self) -> bool:
        if self.is_executing:
            return False
        if self.is_zipped:
            return True
        if self.history_file_path.exists() and self.ranks_file_path.exists():
            return True
        else:
            return False

    @property
    def default_params(self) -> Params:
        try:
            return DEFAULT_PARAMS[self.dataset][self.model]
        except KeyError:
            raise KeyError(
                f"Default parameters not available for model `{self.model.name}` with dataset `{self.dataset.name}`"
            )

    @property
    def ranks(self) -> np.ndarray:
        try:
            # noinspection PyTypeChecker
            return np.load(self.ranks_file_path.resolve().as_posix())
        except Exception as e:
            print(f"Error with {self.name}")
            raise e

    @property
    def losses(self) -> t.Tuple[np.ndarray, np.ndarray]:
        with open(self.history_file_path.as_posix(), 'rb') as file_pi:
            history = pickle.load(file_pi)
        train_loss = history['loss']
        val_loss = history['val_loss']
        return train_loss, val_loss

    @property
    def accuracies(self) -> t.Tuple[np.ndarray, np.ndarray]:
        with open(self.history_file_path.as_posix(), 'rb') as file_pi:
            history = pickle.load(file_pi)
        if 'acc' in history.keys():
            train_acc = history['acc']
            val_acc = history['val_acc']
        else:
            train_acc = history['accuracy']
            val_acc = history['val_accuracy']
        return train_acc, val_acc

    def dump_plots(self):
        # ---------------------------------------------------- 01
        # check if done
        if not self.is_done:
            raise Exception(f"Cannot dump plots as the experiment is not completed ...")

        # ---------------------------------------------------- 02
        # name
        _name = f"{self.dataset.name}-{self.model.name}"

        # ---------------------------------------------------- 03
        # create figures
        avg_rank_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{_name}: Average Rank")
            )
        )
        rank_variance_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{_name}: Rank Variance")
            )
        )
        train_loss_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{_name}: Train Loss")
            )
        )
        val_loss_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{_name}: Validation Loss")
            )
        )

        # ---------------------------------------------------- 04
        # load history and ranks
        with open(self.history_file_path.as_posix(), 'rb') as file_pi:
            history = pickle.load(file_pi)
        # noinspection PyTypeChecker
        ranks = np.load(self.ranks_file_path)

        # ---------------------------------------------------- 05

    def wipe(self):
        if self.is_executing or not self.is_done:
            print(f" > {self.name} ... "
                  f"wiping files ...")
            for _ in self.store_dir.iterdir():
                _.unlink()
            self.store_dir.rmdir()

    @classmethod
    def get_existing_experiments_on_disk(
            cls,
            experiment_type: ExperimentType = None,
            dataset: Dataset = None,
            model: Model = None,
            get_all: bool = False,
    ) -> t.List["Experiment"]:
        _ret = []
        for _type_dir in RESULTS_DIR.iterdir():
            if experiment_type != \
                    ExperimentType[_type_dir.name] and experiment_type is not None:
                continue
            for _dataset_dir in _type_dir.iterdir():
                if dataset != Dataset[_dataset_dir.name] and dataset is not None:
                    continue
                for _model_dir in _dataset_dir.iterdir():
                    if model != Model[_model_dir.name] and model is not None:
                        continue
                    for _id_dir in _model_dir.iterdir():
                        _exp = Experiment(
                            dataset=Dataset[_dataset_dir.name],
                            model=Model[_model_dir.name],
                            id=int(_id_dir.name),
                            type=ExperimentType[_type_dir.name],
                        )
                        if get_all:
                            _ret.append(_exp)
                        else:
                            if _exp.is_done:
                                _ret.append(_exp)
        return _ret

    @classmethod
    def experiment_generator(cls) -> t.Iterable["Experiment"]:
        for _type in EXPERIMENT_TYPES_TO_TRY:
            for _dataset in DATASETS_TO_TRY:
                for _model in MODELS_TO_TRY:
                    for _id in range(NUM_EXPERIMENTS):
                        _exp = Experiment(
                            id=_id, dataset=_dataset, model=_model, type=_type,
                        )
                        try:
                            _ = _exp.default_params
                        except KeyError:
                            continue
                        yield _exp

    @classmethod
    def do_it(cls):
        for _experiment in cls.experiment_generator():
            # ------------------------------------------------ 01
            # if experiment is already complemented then skip
            if _experiment.is_done:
                print(f" > {_experiment.name} ... "
                      f"skipping as already completed ...")
                continue

            # ------------------------------------------------ 02
            # if experiment is executing the skip
            if _experiment.is_executing:
                print(f" > {_experiment.name} ... "
                      f"skipping as someone is executing it ...")
                continue

            # ------------------------------------------------ 03
            # create a semaphore for other threads to detect
            print(f" > {_experiment.name} ... will train and rank ...")
            _experiment.store_dir.mkdir(parents=True, exist_ok=True)
            _experiment.is_executing_file_path.touch(exist_ok=False)

            # ------------------------------------------------ 04
            # get params
            _params = _experiment.default_params

            # ------------------------------------------------ 05
            # get data
            tracesTrain, tracesVal, labelsTrain, labelsVal, X_attack_processed, targets, key_attack = \
                _params.preprocessor.apply(
                    _experiment.dataset,
                    hw_leakage_model=_experiment.model.is_hw_leakage_model)

            # ------------------------------------------------ 06
            # shrink dataset so that it can over-fit easily
            if _experiment.type is ExperimentType.over_fit:
                tracesTrain = tracesTrain[:len(tracesTrain) // 3]
                labelsTrain = labelsTrain[:len(labelsTrain) // 3]

            # ------------------------------------------------ 06
            # get model
            _model_make_fn = _experiment.model.make_fn(_experiment.dataset)
            _model = _model_make_fn(
                input_size=tracesTrain.shape[1], learning_rate=_params.learning_rate
            )
            _model.summary()

            # ------------------------------------------------ 07
            # Ensure the data is in the right shape
            input_layer_shape = _model.get_layer(index=0).input_shape[0]
            if len(input_layer_shape) == 2:
                tracesTrain_shaped = tracesTrain
                tracesVal_shaped = tracesVal
            elif len(input_layer_shape) == 3:
                tracesTrain_shaped = tracesTrain.reshape(
                    (tracesTrain.shape[0], tracesTrain.shape[1], 1))
                tracesVal_shaped = tracesVal.reshape(
                    (tracesVal.shape[0], tracesVal.shape[1], 1))
            else:
                raise Exception(f"Unknown shape {len(input_layer_shape)}")

            # ------------------------------------------------ 08
            # train the model
            print(f" > {_experiment.name} ... training ...")
            _save_best_only = _experiment.type is ExperimentType.early_stopping
            checkpoint = ModelCheckpoint(
                _experiment.model_file_path.as_posix(),
                monitor='val_loss', mode='min', verbose=1,
                save_best_only=_save_best_only)

            if _params.one_cycle_lr:
                print(
                    'During training we will make use of the One Cycle learning rate policy.')
                lr_manager = OneCycleLR(max_lr=_params.learning_rate,
                                        end_percentage=0.2, scale_percentage=0.1,
                                        maximum_momentum=None, minimum_momentum=None,
                                        verbose=False)
                callbacks = [checkpoint, lr_manager]
            else:
                callbacks = [checkpoint]
            _num_classes = 9 if _experiment.model.is_hw_leakage_model else 256
            _epochs = _params.epochs
            # increase epochs when using early stopping
            if _experiment.type is ExperimentType.early_stopping:
                _epochs *= 2
            # call fit
            history = _model.fit(
                x=tracesTrain_shaped,
                y=to_categorical(labelsTrain, num_classes=_num_classes),
                validation_data=(
                tracesVal_shaped, to_categorical(labelsVal, num_classes=_num_classes)),
                batch_size=_params.batch_size, verbose=1, epochs=_epochs,
                callbacks=callbacks)

            # ------------------------------------------------ 09
            # save history
            with open(_experiment.history_file_path, 'wb') as _file:
                pickle.dump(history.history, _file)

            # ------------------------------------------------ 10
            # delete model
            del _model
            del history
            K.clear_session()

            # ------------------------------------------------ 11
            # perform ranking
            # load model
            _model = load_model(_experiment.model_file_path.as_posix())
            # adjust shape
            input_layer_shape = _model.get_layer(index=0).input_shape[0]
            if len(input_layer_shape) == 2:
                tracesAttack_shaped = X_attack_processed
            elif len(input_layer_shape) == 3:
                tracesAttack_shaped = X_attack_processed.reshape(
                    (X_attack_processed.shape[0], X_attack_processed.shape[1], 1))
            else:
                raise Exception(f"Unknown shape {len(input_layer_shape)}")
            # get predictions
            print(f" > {_experiment.name} ... "
                  f"get predictions ...")
            predictions = _model.predict(tracesAttack_shaped, verbose=1)

            print(f" > {_experiment.name} ... "
                  f"computing ranks ...")
            ranks = compute_ranks(
                predictions=predictions,
                all_guess_targets=targets,
                correct_key=key_attack,
                num_attacks=NUM_ATTACKS_PER_EXPERIMENT,
            )

            # Calculate the mean of the rank over the nattack attacks
            avg_rank = np.mean(ranks, axis=0)

            print(np.where(avg_rank <= 0.))

            # ------------------------------------------------ 12
            # save ranks
            np.save(_experiment.ranks_file_path.as_posix(), ranks)

            # ------------------------------------------------ 13
            # as things are over release semaphore
            _experiment.is_executing_file_path.unlink()
            _experiment.model_file_path.unlink()

    # noinspection PyArgumentList
    @classmethod
    def report_it_make_simple_plots(
            cls, _results: t.Dict[str, np.ndarray], _plot_dir: pathlib.Path,
            _fig_name: str, _rank_plot_until: int
    ):

        # ------------------------------------------------------ 01
        # get data
        avg_ranks = _results['avg_ranks']
        rank_variances = _results['rank_variances']
        train_losses = _results['train_losses']
        val_losses = _results['val_losses']
        train_accuracies = _results['train_accuracies']
        val_accuracies = _results['val_accuracies']
        traces_needed_for_rank_0s = _results['traces_needed_for_rank_0s']
        num_experiments = len(avg_ranks)

        # ------------------------------------------------------ 02
        # make figures
        _avg_rank_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(
                    text=f"Average Rank: {_fig_name}")
            )
        )
        # _rank_variance_fig = go.Figure(
        #     layout=go.Layout(
        #         title=go.layout.Title(
        #             text=f"Rank Variance: {_fig_name}")
        #     )
        # )
        _train_loss_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"Train Loss: {_fig_name}")
            )
        )
        _val_loss_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"Validation Loss: {_fig_name}")
            )
        )
        _train_acc_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"Train Accuracy: {_fig_name}")
            )
        )
        _val_acc_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"Validation Accuracy: {_fig_name}")
            )
        )

        # ------------------------------------------------------ 03
        # # update y range for some figures
        # _avg_rank_fig.update_layout(
        #     yaxis_range=[avg_ranks.min(), avg_ranks.max()])
        # # _rank_variance_fig.update_layout(
        # #     yaxis_range=[rank_variances.min(), rank_variances.max()])
        # _train_loss_fig.update_layout(
        #     yaxis_range=[train_losses.min(), train_losses.max()])
        # _val_loss_fig.update_layout(
        #     yaxis_range=[val_losses.min(), val_losses.max()])
        # _train_acc_fig.update_layout(
        #     yaxis_range=[train_accuracies.min(), train_accuracies.max()])
        # _val_acc_fig.update_layout(
        #     yaxis_range=[val_accuracies.min(), val_accuracies.max()])

        # ------------------------------------------------------ 04
        # plot to figures
        for _exp_id in range(num_experiments):
            _avg_rank_fig.add_trace(
                go.Scatter(
                    x=np.arange(_rank_plot_until),
                    y=avg_ranks[_exp_id, :_rank_plot_until],
                    mode='lines',
                    name=f"exp_{_exp_id:03d}",
                    showlegend=False,
                )
            )
            # _rank_variance_fig.add_trace(
            #     go.Scatter(
            #         x=np.arange(_rank_plot_until),
            #         y=rank_variances[_exp_id, :_rank_plot_until],
            #         mode='lines',
            #         name=f"exp_{_exp_id:03d}",
            #         showlegend=False,
            #     )
            # )
            _train_loss_fig.add_trace(
                go.Scatter(
                    x=np.arange(len(train_losses[_exp_id])),
                    y=train_losses[_exp_id],
                    mode='lines',
                    name=f"exp_{_exp_id:03d}",
                    showlegend=False,
                )
            )
            _val_loss_fig.add_trace(
                go.Scatter(
                    x=np.arange(len(val_losses[_exp_id])),
                    y=val_losses[_exp_id],
                    mode='lines',
                    name=f"exp_{_exp_id:03d}",
                    showlegend=False,
                )
            )
            _train_acc_fig.add_trace(
                go.Scatter(
                    x=np.arange(len(train_accuracies[_exp_id])),
                    y=train_accuracies[_exp_id],
                    mode='lines',
                    name=f"exp_{_exp_id:03d}",
                    showlegend=False,
                )
            )
            _val_acc_fig.add_trace(
                go.Scatter(
                    x=np.arange(len(val_accuracies[_exp_id])),
                    y=val_accuracies[_exp_id],
                    mode='lines',
                    name=f"exp_{_exp_id:03d}",
                    showlegend=False,
                )
            )

        # ----------------------------------------------- 02.05
        # save figures
        fig_update_layout(_avg_rank_fig)
        # fig_update_layout(_rank_variance_fig)
        fig_update_layout(_train_loss_fig)
        fig_update_layout(_val_loss_fig)
        fig_update_layout(_train_acc_fig)
        fig_update_layout(_val_acc_fig)
        _avg_rank_fig.write_image(
            (_plot_dir / f"average_rank.svg").as_posix(), engine='kaleido')
        # _rank_variance_fig.write_image(
        #     (_plot_dir / f"rank_variance.svg").as_posix(), engine='kaleido')
        _train_loss_fig.write_image(
            (_plot_dir / f"train_loss.svg").as_posix(), engine='kaleido')
        _val_loss_fig.write_image(
            (_plot_dir / f"val_loss.svg").as_posix(), engine='kaleido')
        _train_acc_fig.write_image(
            (_plot_dir / f"train_acc.svg").as_posix(), engine='kaleido')
        _val_acc_fig.write_image(
            (_plot_dir / f"val_acc.svg").as_posix(), engine='kaleido')

    @classmethod
    def report_it_group_experiments(
            cls, _experiments: t.List["Experiment"], is_mcovc: bool
    ) -> t.Dict[str, np.ndarray]:

        _avg_ranks = []
        _rank_variances = []
        _train_losses = []
        _val_losses = []
        _train_accuracies = []
        _val_accuracies = []
        _traces_needed_for_rank_0s = []

        for _experiment in _experiments:

            _ranks = _experiment.ranks
            _avg_rank = np.mean(_ranks, axis=0)
            _rank_variance = np.var(_ranks, axis=0)
            _train_loss, _val_loss = _experiment.losses
            _train_acc, _val_acc = _experiment.accuracies
            if is_mcovc:
                _traces_needed_for_rank_0 = np.nan
            else:
                _traces_needed_for_rank_0 = np.where(_avg_rank <= 0.0)[0]
                if len(_traces_needed_for_rank_0) == 0:
                    _traces_needed_for_rank_0 = np.inf
                else:
                    _traces_needed_for_rank_0 = _traces_needed_for_rank_0[0]

            _avg_ranks.append(_avg_rank)
            _rank_variances.append(_rank_variance)
            _train_losses.append(_train_loss)
            _val_losses.append(_val_loss)
            _train_accuracies.append(_train_acc)
            _val_accuracies.append(_val_acc)
            _traces_needed_for_rank_0s.append(_traces_needed_for_rank_0)

        return dict(
            avg_ranks=np.asarray(_avg_ranks),
            rank_variances=np.asarray(_rank_variances),
            train_losses=np.asarray(_train_losses),
            val_losses=np.asarray(_val_losses),
            train_accuracies=np.asarray(_train_accuracies),
            val_accuracies=np.asarray(_val_accuracies),
            traces_needed_for_rank_0s=np.asarray(_traces_needed_for_rank_0s),
        )

    @classmethod
    def report_it_violin_avg_ranks(
            cls,
            _report_df_detailed: pd.DataFrame,
            _violin_fig_dir: pathlib.Path,
            _traces_to_use_for_avg_rank_es: t.List[int]
    ):
        # loop over 6 values
        for i in range(6):

            # ----------------------------------------------------- 01
            # traces_to_use_for_avg_rank
            _traces_to_use_for_avg_rank = _traces_to_use_for_avg_rank_es[i]
            # file path
            _file_path = _violin_fig_dir / f"violin_avg_rank_{i}.svg"

            # ----------------------------------------------------- 02
            # make figure
            _violin_fig = px.violin(
                _report_df_detailed,
                y=f"avg_rank_{i}",
                x="model",
                color="model",
                box=False,
                points="all",
                hover_data=_report_df_detailed.columns,
                labels={
                    f'avg_rank_{i}': 'Average Ranks'
                },
                title=f"Distribution of average ranks over {NUM_EXPERIMENTS} "
                      f"experiments (with {_traces_to_use_for_avg_rank} traces)",
            )

            # ----------------------------------------------------- 03
            # add text annotation
            # ----------------------------------------------------- 03.01
            # compute y offset
            _all_avg_ranks = _report_df_detailed[f"avg_rank_{i}"]
            _y_offset = (_all_avg_ranks.max() - _all_avg_ranks.min()) * 0.3
            _y_offset += _all_avg_ranks.max()
            # ----------------------------------------------------- 03.02
            # loop over models
            for _model_name in _report_df_detailed.model.unique():
                # ------------------------------------------------- 03.03
                # get avg ranks
                _avg_ranks = _report_df_detailed[
                    _report_df_detailed.model == _model_name
                ][f"avg_rank_{i}"]
                # ------------------------------------------------- 03.04
                # create visual elements
                _min = _avg_ranks.min()
                _max = _avg_ranks.max()
                _median = _avg_ranks.median()
                _mean = _avg_ranks.mean()
                _bgcolor = 'lightgrey'
                _bordercolor = 'black'
                _text = f"<br>min: {_min:.2f}" \
                        f"<br>max: {_max:.2f}" \
                        f"<br>median: {_median:.2f}" \
                        f"<br>mean: {_mean:.2f}"
                # ------------------------------------------------- 03.05
                # add annotations to figure
                _violin_fig.add_annotation(
                    x=_model_name,
                    y=_y_offset,
                    text=_text,
                    bgcolor=_bgcolor,
                    bordercolor=_bordercolor,
                    showarrow=False,
                )

            # ----------------------------------------------------- 04
            # write violin plot
            fig_update_layout(_violin_fig)
            _violin_fig.write_image(_file_path.as_posix(), engine='kaleido')

    @classmethod
    def report_it_violin(
            cls,
            _report_df_detailed: pd.DataFrame,
            _report_df: pd.DataFrame,
            _violin_fig_path: pathlib.Path
    ):
        # ------------------------------------------------- 01
        # make violin figure
        _violin_fig = px.violin(
            _report_df_detailed,
            y="min traces needed for average rank to be zero",
            x="model",
            color="model",
            box=False,
            points="all",
            hover_data=_report_df_detailed.columns,
            title=f"Distribution of min traces needed for average rank to be zero over "
                  f"{NUM_EXPERIMENTS} experiments",
        )

        # ------------------------------------------------- 02
        # compute info for annotations
        _all_annotation_data = {}
        # ------------------------------------------------- 02.01
        # loop over all models
        for _model in MODELS_TO_TRY:

            # filter down
            _filter_report_df = _report_df[_report_df.model == _model.name]
            if len(_filter_report_df) == 0:
                continue
            assert len(_filter_report_df) == 1
            _filter_report_df = _filter_report_df.iloc[0]

            # not applicable for mcovc models
            if _model.is_mcovc:
                _text = f" Not available "
                _bgcolor = 'lightgrey'
                _bordercolor = 'black'
                _text += f"<br>min: NA" \
                         f"<br>max: NA" \
                         f"<br>median: NA" \
                         f"<br>mean: NA"

            # add info for remaining
            else:

                if _filter_report_df.failed_percentage == 0.:
                    _text = f" All passed "
                    _bgcolor = 'lightgreen'
                    _bordercolor = 'green'
                else:
                    _text = f" {_filter_report_df.failed_percentage:.2f} % failed "
                    _bgcolor = 'pink'
                    _bordercolor = 'red'

                _min = _filter_report_df.traces_needed_range_min
                if _min != np.inf:
                    _min = int(_min)
                _max = _filter_report_df.traces_needed_range_max
                if _max != np.inf:
                    _max = int(_max)
                _median = _filter_report_df.traces_needed_range_median
                _mean = _filter_report_df.traces_needed_range_mean
                _text += f"<br>min: {_min}" \
                         f"<br>max: {_max}" \
                         f"<br>median: {_median}" \
                         f"<br>mean: {_mean}"

            # annotation settings
            _all_annotation_data[_model] = dict(
                text=_text, bordercolor=_bordercolor, bgcolor=_bgcolor,
            )

        # ------------------------------------------------- 02.02
        # compute y offset
        _max = _report_df.traces_needed_range_max.replace(np.inf, np.nan).max()
        _min = _report_df.traces_needed_range_min.min()
        _y_offset = (_max - _min) * 0.3
        _y_offset += _max

        # ------------------------------------------------- 03
        # add annotations to figure
        for _model, _annotation_data in _all_annotation_data.items():
            _violin_fig.add_annotation(
                x=_model.name,
                y=_y_offset,
                text=_annotation_data['text'],
                bgcolor=_annotation_data['bgcolor'],
                bordercolor=_annotation_data['bordercolor'],
                showarrow=False,
            )

        # ------------------------------------------------- 04
        # write figure
        fig_update_layout(_violin_fig)
        _violin_fig.write_image(_violin_fig_path.as_posix(), engine='kaleido')

    # noinspection PyArgumentList
    @classmethod
    def report_it_make_md_files(cls, _all_report_dfs: t.Dict):

        # --------------------------------------------------- 01
        # base report md file
        _base_report_md_file_path = ROOT_DIR / "reports.md"
        _md_lines = [
            "",
            f"# Detailed analysis for below datasets is available: "
            "",
            "",
        ]
        for _dataset in DATASETS_TO_TRY:
            _md_lines.append(
                f"+ [{_dataset.name}](reports/{_dataset.name}.md)"
            )
        _base_report_md_file_path.write_text("\n".join(_md_lines))

        # --------------------------------------------------- 02
        # loop over datasets
        for _dataset in DATASETS_TO_TRY:

            # ----------------------------------------------- 10.01
            _md_lines = [
                "",
                f"# Detailed analysis for dataset `{_dataset.name}` ...",
                "",
            ]

            # ----------------------------------------------- 10.02
            for _type in EXPERIMENT_TYPES_TO_TRY:

                # ------------------------------------------- 10.02.01
                _md_lines += [
                    "",
                    f"## Train and attack results for `{_type.name}` ",
                    "",
                ]

                # ------------------------------------------- 10.02.02
                _md_lines += [
                    "",
                    f"### Number of traces needed for average rank to be zero ",
                    "",
                    "![Number of traces needed for average rank to be zero]"
                    f"(../plots/{_type.name}/{_dataset.name}/violin.svg)",
                    "",
                ]

                # ------------------------------------------- 10.02.03
                _md_lines += [
                    "",
                    f"### Average ranks over given number of traces ",
                    "",
                    "![violin_avg_rank_0]"
                    f"(../plots/{_type.name}/{_dataset.name}/violin_avg_rank_0.svg)",
                    "![violin_avg_rank_1]"
                    f"(../plots/{_type.name}/{_dataset.name}/violin_avg_rank_1.svg)",
                    "![violin_avg_rank_2]"
                    f"(../plots/{_type.name}/{_dataset.name}/violin_avg_rank_2.svg)",
                    "![violin_avg_rank_3]"
                    f"(../plots/{_type.name}/{_dataset.name}/violin_avg_rank_3.svg)",
                    "![violin_avg_rank_4]"
                    f"(../plots/{_type.name}/{_dataset.name}/violin_avg_rank_4.svg)",
                    "![violin_avg_rank_5]"
                    f"(../plots/{_type.name}/{_dataset.name}/violin_avg_rank_5.svg)",
                ]

                # ------------------------------------------- 10.02.04
                _md_lines += [
                    "",
                    f"### Average ranks, train/val loss and train/val accuracy ",
                    "",
                ]
                _table_header = "|"
                _table_sep = "|"
                _table_avg_rank = "|"
                # _table_rank_variance = "|"
                _table_train_loss = "|"
                _table_val_loss = "|"
                _table_train_acc = "|"
                _table_val_acc = "|"
                # ------------------------------------------- 10.02.04.01
                # loop over models
                for _model in MODELS_TO_TRY:

                    # get report df
                    _report_df = _all_report_dfs[_type][_dataset]
                    _filter_report_df = _report_df[_report_df.model == _model.name]
                    if len(_filter_report_df) == 0:
                        continue
                    assert len(_filter_report_df) == 1
                    _filter_report_df = _filter_report_df.iloc[0]
                    _failed_percentage = _filter_report_df.failed_percentage

                    # make success status
                    if _failed_percentage == np.nan:
                        _table_success_status = \
                            f"<span style='color:blue'> " \
                            f"**IS MULTI-CHANNEL** " \
                            f"</span>"
                    else:
                        if _failed_percentage > 0.:
                            _table_success_status = \
                                f"<span style='color:red'> " \
                                f"**{_failed_percentage:.2f} % FAILED** " \
                                f"</span>"
                        else:
                            _table_success_status = \
                                f"<span style='color:green'>" \
                                f"**ALL PASSED** " \
                                f"</span>"

                    # update table
                    _plot_relative_path = \
                        f"../plots/{_type.name}/{_dataset.name}/{_model.name}"
                    _table_header += f"{_model.name}<br>{_table_success_status}|"
                    _table_sep += "---|"
                    _table_avg_rank += \
                        f"![Average Rank]({_plot_relative_path}/average_rank.svg)|"
                    # _table_rank_variance +=
                    #    f"![Rank Variance]({_plot_relative_path}/rank_variance.svg)|"
                    _table_train_loss += \
                        f"![Train Loss]({_plot_relative_path}/train_loss.svg)|"
                    _table_val_loss += \
                        f"![Validation Loss]({_plot_relative_path}/val_loss.svg)|"
                    _table_train_acc += \
                        f"![Train Accuracy]({_plot_relative_path}/train_acc.svg)|"
                    _table_val_acc += \
                        f"![Validation Accuracy]({_plot_relative_path}/val_acc.svg)|"
                # ------------------------------------------- 10.02.04.02
                _md_lines += [
                    "",
                    _table_header, _table_sep,
                    _table_avg_rank,
                    # _table_rank_variance,
                    _table_train_loss, _table_val_loss,
                    _table_train_acc, _table_val_acc,
                    "",
                ]

            # ----------------------------------------------- 10.03
            _report_md_file_path = ROOT_DIR / "reports" / f"{_dataset.name}.md"
            _report_md_file_path.write_text(
                "\n".join(_md_lines)
            )

    # noinspection PyArgumentList
    @classmethod
    def report_it(cls):

        # --------------------------------------------------- 01
        # loop over - no early stopping
        print("Generating report ...")
        _all_report_dfs = {}

        # --------------------------------------------------- 02
        # loop over datasets
        # loop over
        for _type, _dataset in itertools.product(
                EXPERIMENT_TYPES_TO_TRY,
                DATASETS_TO_TRY,
        ):
            # ----------------------------------------------- 03
            # log
            print("")
            print(f"Generating reports for {_type.name}-{_dataset.name}")

            # ----------------------------------------------- 04
            # loop over models
            _all_avg_ranks = {}
            _all_traces_needed_for_rank_0s = {}
            print(" >> Looping over all models")
            for _, _model in enumerate(MODELS_TO_TRY):
                print(f"   [{_+1}] model: {_model.name}")

                # ------------------------------------------- 04.01
                # get experiments
                print("      - reading experiments")
                _experiments = cls.get_existing_experiments_on_disk(
                    experiment_type=_type, dataset=_dataset, model=_model,
                )
                # if no experiments skip
                if not bool(_experiments):
                    continue

                # ------------------------------------------- 04.02
                # group results from all experiments
                print("      - grouping experiments")
                _results = cls.report_it_group_experiments(
                    _experiments, _model.is_mcovc)
                _all_avg_ranks[_model] = _results['avg_ranks']
                _all_traces_needed_for_rank_0s[_model] = \
                    _results['traces_needed_for_rank_0s']
                _num_experiments = len(_experiments)

                # ------------------------------------------- 04.03
                # make simple plots
                print("      - make simple plots")
                _rank_plot_until = _dataset.rank_plot_until
                _fig_name = f"{_dataset.name}-{_model.name} ({_type.name})"
                _plot_relative_path = f"{_type.name}/{_dataset.name}/{_model.name}"
                _plot_dir = PLOTS_DIR / _plot_relative_path
                if not _plot_dir.exists():
                    _plot_dir.mkdir(parents=True)
                cls.report_it_make_simple_plots(
                    _results, _plot_dir, _fig_name, _rank_plot_until)

            # ----------------------------------------------- 05
            # make data frame for violin fig data
            _report_df_detailed = pd.DataFrame()
            _traces_to_use_for_avg_rank = _dataset.traces_to_use_for_rank()
            for _model in _all_avg_ranks.keys():
                _avg_ranks = _all_avg_ranks[_model]
                _traces_needed_for_rank_0s = _all_traces_needed_for_rank_0s[_model]
                for _ in range(NUM_EXPERIMENTS):
                    _report_df_detailed = _report_df_detailed.append(
                        {
                            'model': _model.name,
                            'experiment id': _,
                            'min traces needed for average rank to be zero':
                                _traces_needed_for_rank_0s[_],
                            'avg_rank_0': _avg_ranks[_, _traces_to_use_for_avg_rank[0]],
                            'avg_rank_1': _avg_ranks[_, _traces_to_use_for_avg_rank[1]],
                            'avg_rank_2': _avg_ranks[_, _traces_to_use_for_avg_rank[2]],
                            'avg_rank_3': _avg_ranks[_, _traces_to_use_for_avg_rank[3]],
                            'avg_rank_4': _avg_ranks[_, _traces_to_use_for_avg_rank[4]],
                            'avg_rank_5': _avg_ranks[_, _traces_to_use_for_avg_rank[5]],
                        }, ignore_index=True,
                    )

            # ----------------------------------------------- 06
            # update report data frame for markdown files
            _report_df = pd.DataFrame()
            # loop over models
            for _model in _all_traces_needed_for_rank_0s.keys():
                _traces_needed_for_rank_0s = _all_traces_needed_for_rank_0s[_model]
                _total_experiments = len(_traces_needed_for_rank_0s)
                if _model.is_mcovc:
                    _failed_experiments = np.nan
                    _failed_percentage = np.nan
                else:
                    _failed_experiments = (_traces_needed_for_rank_0s == np.inf).sum()
                    _failed_percentage = \
                        float(_failed_experiments / _total_experiments) * 100.
                _traces_needed_range_min = _traces_needed_for_rank_0s.min()
                _traces_needed_range_max = _traces_needed_for_rank_0s.max()
                _traces_needed_range_median = np.median(_traces_needed_for_rank_0s)
                _traces_needed_range_mean = np.mean(_traces_needed_for_rank_0s)
                _report_df = _report_df.append(
                    {
                        "model": _model.name,
                        "failed_experiments": _failed_experiments,
                        "total_experiments": _total_experiments,
                        "failed_percentage": _failed_percentage,
                        "traces_needed_range_min": _traces_needed_range_min,
                        "traces_needed_range_max": _traces_needed_range_max,
                        "traces_needed_range_median": _traces_needed_range_median,
                        "traces_needed_range_mean": _traces_needed_range_mean,
                    }, ignore_index=True,
                )
            # back it up
            if _type not in _all_report_dfs.keys():
                _all_report_dfs[_type] = {}
            if _dataset not in _all_report_dfs[_type].keys():
                _all_report_dfs[_type][_dataset] = _report_df

            # ----------------------------------------------- 07
            # make violin plot for num traces needed for rank zero
            print(" >> Making violin plot")
            _violin_fig_path = PLOTS_DIR / _type.name / _dataset.name / "violin.svg"
            cls.report_it_violin(_report_df_detailed, _report_df, _violin_fig_path)

            # ----------------------------------------------- 08
            # make violin plots for avg ranks across six num traces used
            print(" >> Making average ranks violin plot")
            _violin_fig_dir = PLOTS_DIR / _type.name / _dataset.name
            cls.report_it_violin_avg_ranks(
                _report_df_detailed,
                _violin_fig_dir,
                _dataset.traces_to_use_for_rank())

        # --------------------------------------------------- 09
        # make report md files
        print(" >> Making report md files")
        cls.report_it_make_md_files(_all_report_dfs)

    @classmethod
    def wipe_it(cls):
        for _e in cls.get_existing_experiments_on_disk(get_all=True):
            _e.wipe()

    @classmethod
    def zip_it(cls):
        # get experiments on disk
        _existing_experiments_on_disk = cls.get_existing_experiments_on_disk(
            get_all=False)

        # if any of the experiment is already zipped then raise error
        for _exp in _existing_experiments_on_disk:
            if _exp.is_zipped:
                raise Exception(
                    f"Experiment {_exp.name} is already zipped ..."
                )

        # make and/or load zip
        _zip_file_path = ROOT_DIR / "results.zip"
        if not _zip_file_path.exists():
            _zip_file = zipfile.ZipFile(_zip_file_path, 'w', zipfile.ZIP_BZIP2)
            _zip_file.close()
        _zip_file = zipfile.ZipFile(_zip_file_path, 'a', zipfile.ZIP_BZIP2)

        # zip it
        for _experiment in tqdm.tqdm(_existing_experiments_on_disk, desc="zipping"):

            # log
            # print(f"Zipping {_experiment.name} ...")

            # write to archive
            _history_archive_name = "/".join(_experiment.history_file_path.parts[-6:])
            _ranks_archive_name = "/".join(_experiment.ranks_file_path.parts[-6:])
            _zip_file.write(
                _experiment.history_file_path, arcname=_history_archive_name
            )
            _zip_file.write(
                _experiment.ranks_file_path, arcname=_ranks_archive_name
            )

            # delete files
            for _ in _experiment.store_dir.iterdir():
                _.unlink()
            _experiment.store_dir.rmdir()

        # close zip finally
        _zip_file.close()

    @classmethod
    def unzip_it(cls):

        _zip_file_path = ROOT_DIR / "results.zip"
        if not _zip_file_path.exists():
            raise Exception(
                f"Cannot file results.zip file on disk"
            )
        if RESULTS_DIR.exists():
            raise Exception(
                f"We expect results dir should not be present while unzipping"
            )
        _zip_file = zipfile.ZipFile(_zip_file_path, 'r', zipfile.ZIP_BZIP2)
        _zip_file.extractall(path=RESULTS_DIR.parent)


def main():
    print(
        "*******************************************************************************")
    print("        ", sys.argv)
    print("        results dir: ", RESULTS_DIR.resolve().as_posix())
    print(
        "*******************************************************************************")

    _mode = sys.argv[1]
    if _mode == 'do_it':
        Experiment.do_it()
    elif _mode == 'report_it':
        Experiment.report_it()
    elif _mode == 'wipe_it':
        Experiment.wipe_it()
    elif _mode == 'zip_it':
        Experiment.zip_it()
    elif _mode == 'unzip_it':
        Experiment.unzip_it()
    else:
        raise Exception(f"Unknown {_mode}")

    print(
        "*******************************************************************************")
    print(
        "*******************   The requested job is now completed   ********************")
    print(
        "*******************************************************************************")

    print()
    print()
    print()


def _filter_experiments():
    _es = Experiment.get_existing_experiments_on_disk(
        experiment_type=ExperimentType.original,
        dataset=Dataset.ascad_v1_vk_0_noisy,
        model=Model.aisy_mlp_id,
    )
    for _e in _es:
        _mean_rank = np.mean(_e.ranks, axis=0)
        _where = np.where(_mean_rank <= 0.)[0][0]
        print(_e.id, _where)


if __name__ == '__main__':
    # _filter_experiments()
    main()
