import inspect
import json
import logging
from typing import List, Tuple, Optional

import numpy as np
from pathlib import Path

from .base_dataset import BaseDataset
from .corruption import apply_corruption, select_corrupt_nodes


class TimeSeriesForecastingDataset(BaseDataset):
    """
    A dataset class for time series forecasting problems, handling the loading, parsing, and partitioning
    of time series data into training, validation, and testing sets based on provided ratios.
    
    This class supports configurations where sequences may or may not overlap, accommodating scenarios
    where time series data is drawn from continuous periods or distinct episodes, affecting how
    the data is split into batches for model training or evaluation.
    
    Attributes:
        data_file_path (str): Path to the file containing the time series data.
        description_file_path (str): Path to the JSON file containing the description of the dataset.
        data (np.ndarray): The loaded time series data array, split according to the specified mode.
        description (dict): Metadata about the dataset, such as shape and other properties.
    """

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int,
                 output_len: int, memmap: bool = False, overlap: bool = False,
                 data_range: Tuple[int, int] = None, is_noise: bool = False,
                 logger: logging.Logger = None, long_history_len: int = None,
                 node_indices: Optional[List[int]] = None,
                 corruption: Optional[dict] = None) -> None:
        """
        Initializes the TimeSeriesForecastingDataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset.
            train_val_test_ratio (List[float]): Ratios for splitting the dataset into train, validation, and test sets.
                Each value should be a float between 0 and 1, and their sum should ideally be 1.
            mode (str): The operation mode of the dataset. Valid values are 'train', 'valid', or 'test'.
            input_len (int): The length of the input sequence (number of historical points).
            output_len (int): The length of the output sequence (number of future points to predict).
            overlap (bool): Flag to determine if training/validation/test splits should overlap. 
                Defaults to False for strictly non-overlapping periods. Set to True to allow overlap.
            logger (logging.Logger): logger.

        Raises:
            AssertionError: If `mode` is not one of ['train', 'valid', 'test'].
        """
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(dataset_name, train_val_test_ratio, mode, memmap)
        self.input_len = input_len
        self.output_len = output_len
        self.overlap = overlap
        self.logger = logger
        self.data_range = data_range
        self.is_noise = is_noise
        self.long_history_len = long_history_len
        self.node_indices = node_indices
        self.corruption = corruption
        self.corrupt_nodes = None  # set in _load_data if corruption is applied

        if self.is_noise:
            print(f"Loading noisy data from {dataset_name}/noise")
            self.data_file_path = f'datasets/{dataset_name}/noise/data.dat'
            self.description_file_path = f'datasets/{dataset_name}/noise/desc.json'
        else:
            print(f"Loading data from {dataset_name}")
            self.data_file_path = f'datasets/{dataset_name}/data.dat'
            self.description_file_path = f'datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
        self.data = self._load_data()

    def _load_description(self) -> dict:
        """
        Loads the description of the dataset from a JSON file.

        Returns:
            dict: A dictionary containing metadata about the dataset, such as its shape and other properties.

        Raises:
            FileNotFoundError: If the description file is not found.
            json.JSONDecodeError: If there is an error decoding the JSON data.
        """

        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {self.description_file_path}') from e

    def _load_data(self) -> np.ndarray:
        """
        Loads the time series data from a file and splits it according to the selected mode.

        Returns:
            np.ndarray: The data array for the specified mode (train, validation, or test).

        Raises:
            ValueError: If there is an issue with loading the data file or if the data shape is not as expected.
        """

        try:
            data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=tuple(self.description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_file_path}') from e

        if self.data_range is not None:
            data = data[self.data_range[0]:self.data_range[1]]

        # Apply synthetic corruption before split (same corruption across train/val/test)
        if self.corruption is not None:
            data = apply_corruption(data, self.corruption)
            self.corrupt_nodes = select_corrupt_nodes(
                data.shape[1], self.corruption['rate'],
                self.corruption.get('seed', 42),
                self.corruption.get('exclude_nodes', None)
            )

        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        # Automatically configure the overlap parameter
        minimal_len = self.input_len + self.output_len
        if minimal_len > {'train': train_len, 'valid': valid_len, 'test': test_len}[self.mode]:
            self.overlap = True  # Enable overlap when the train, validation, or test set is too short
            current_frame = inspect.currentframe()
            file_name = inspect.getfile(current_frame)
            line_number = current_frame.f_lineno - 7
            dataset = {'train': 'Training', 'valid': 'Validation', 'test': 'Test'}[self.mode]
            if self.logger is not None:
                self.logger.info(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')
            else:
                print(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')

        if self.mode == 'train':
            offset = self.output_len if self.overlap else 0
            seg = data[:train_len + offset]
        elif self.mode == 'valid':
            offset_left = self.input_len - 1 if self.overlap else 0
            offset_right = self.output_len if self.overlap else 0
            seg = data[train_len - offset_left : train_len + valid_len + offset_right]
        elif self.mode == 'test':
            offset = self.input_len - 1 if self.overlap else 0
            seg = data[train_len + valid_len - offset:]
        elif self.mode == 'all':
            offset = self.output_len if self.overlap else 0
            seg = data[:train_len + valid_len + test_len + offset]
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of ['train', 'valid', 'test', 'all'].")

        if not self.memmap:
            seg = seg.copy()

        # Filter nodes if node_indices is specified
        if self.node_indices is not None:
            seg = seg[:, self.node_indices, :]

        return seg

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing 'inputs', 'target', and 'index', where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively. The 'index' is the sample index.
                  Optionally includes 'long_history' if long_history_len is set.
        """
        history_data = self.data[index:index + self.input_len]
        future_data = self.data[index + self.input_len:index + self.input_len + self.output_len]
        if self.memmap:
            history_data = history_data.copy()
            future_data = future_data.copy()

        result = {'inputs': history_data, 'target': future_data, 'index': index}

        # Add long history if configured (for models like STEP)
        if self.long_history_len is not None:
            long_start = index + self.input_len - self.long_history_len
            if long_start >= 0:
                long_history = self.data[long_start:index + self.input_len]
            else:
                # Pad with zeros if not enough history
                pad_len = -long_start
                long_history = np.concatenate([
                    np.zeros((pad_len,) + self.data.shape[1:], dtype=self.data.dtype),
                    self.data[0:index + self.input_len]
                ], axis=0)
            if self.memmap:
                long_history = long_history.copy()
            result['long_history'] = long_history

        return result

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return len(self.data) - self.input_len - self.output_len + 1

    def add_noise(self, noisy_nodes: List[int], noise_ratio: float, seed: int = 42, 
                target_columns: Optional[List[int]] = None,
                save_dir: Optional[str] = None) -> None:
        """
        원본 데이터에 노이즈 추가. save_dir이 존재하면 로드, 없으면 생성 후 저장.
        
        Args:
            noisy_nodes: 노이즈를 추가할 노드 인덱스 리스트
            noise_ratio: 노이즈 비율 (std = ratio * data_std)
            seed: 랜덤 시드
            target_columns: 노이즈를 추가할 컬럼 인덱스 리스트 (None이면 전체 컬럼)
            save_dir: 저장/로드할 디렉토리 (None이면 저장 안 함)
        """
        if target_columns is None:
            target_columns = list(range(self.data.shape[-1]))
        
        # 이미 존재하면 로드
        if save_dir is not None and Path(save_dir).exists():
            self._load_noisy_data(save_dir)
            print(f"Loaded existing noisy data from {save_dir}")
            return
        
        # 새로 생성
        if isinstance(self.data, np.memmap):
            self.data = np.array(self.data)
        
        rng = np.random.default_rng(seed)
        
        data_subset = self.data[:, noisy_nodes, :][:, :, target_columns]
        data_std = np.std(data_subset)
        noise_std = noise_ratio * data_std
        
        noise_shape = (self.data.shape[0], len(noisy_nodes), len(target_columns))
        noise = rng.standard_normal(noise_shape) * noise_std
        
        self.data[np.ix_(range(self.data.shape[0]), noisy_nodes, target_columns)] += noise.astype(self.data.dtype)
        
        # 저장
        if save_dir is not None:
            self._save_noisy_data(save_dir, noisy_nodes, noise_ratio, seed, target_columns)
            print(f"Created and saved noisy data to {save_dir}")



    def _load_noisy_data(self, save_dir: str) -> None:
        """저장된 노이즈 데이터 로드."""
        save_path = Path(save_dir)
        
        with open(save_path / "desc.json", 'r') as f:
            desc = json.load(f)
        
        self.data = np.memmap(
            save_path / "data.dat",
            dtype=desc["dtype"],
            mode='r',
            shape=tuple(desc["shape"])
        )


    def _save_noisy_data(self, save_dir: str, noisy_nodes: List[int], 
                        noise_ratio: float, seed: int, target_columns: List[int]) -> None:
        """노이즈가 추가된 데이터 저장."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # data.dat 저장
        data_path = save_path / "data.dat"
        fp = np.memmap(data_path, dtype=self.data.dtype, mode='w+', shape=self.data.shape)
        fp[:] = self.data[:]
        fp.flush()
        del fp
        
        # desc.json 저장
        desc = {
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype),
            "noise": {
                "noisy_nodes": noisy_nodes,
                "noise_ratio": noise_ratio,
                "seed": seed,
                "target_columns": target_columns
            }
        }
        
        with open(save_path / "desc.json", 'w') as f:
            json.dump(desc, f, indent=2)