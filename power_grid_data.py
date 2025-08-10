import os
import zipfile
import os.path as osp
from typing import Callable, List, Optional
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
import numpy as np
import h5py
import csv
import gdown

def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]

def preformat_Powergrid(dataset_dir, train_dataset, test_dataset):
    """Load and preformat Power Grid datasets.

    Returns:
        PyG dataset object
    """

    dataset = join_dataset_splits(
        [Powergrid(root=dataset_dir, split=split, train_dataset=train_dataset, test_dataset=test_dataset)
         for split in ['train', 'valid', 'test']]
    )
    return dataset


def get_length_of_dataset(grid_path):
    count = 0
    for file in sorted(os.listdir(grid_path)):
        if file.startswith('grid_data_'):
            if count == 0:
                startIndex = int(os.path.splitext(
                    file)[0].split('grid_data_')[1])
                digits = (os.path.splitext(
                    file)[0].split('grid_data_')[1])
            count += 1
    return count, startIndex, digits


def convert_binary_array_to_index(binary_array):
    length_input = len(binary_array)
    new_array = []
    for i in range(length_input):
        if binary_array == True:
            new_array.append(i)
    return new_array



class Powergrid(InMemoryDataset):
    FILE_ID = '1yEZVwvaGenQ_yJvAPRXmNVzVEP-Tnmgz'
    URL = f'https://drive.google.com/uc?id={FILE_ID}'
    MARKER_FILE = 'extracted.marker'

    def __init__(
        self,

        root: str,
        split: str = 'train',
        train_dataset: str = '',
        test_dataset: str = '',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        if split == 'val':
            split = 'valid'
        assert split in ['train', 'valid', 'test'], split
        self.split = split
        self.root2 = root
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.task = "snbs"
        super().__init__(root, transform, pre_transform, pre_filter)
        path3 = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path3)
        # Ensure node features are float tensors for downstream models
        if hasattr(self.data, 'x'):
            self.data.x = self.data.x.to(torch.float)
        # Ensure target values use float tensors to avoid dtype mismatches
        if hasattr(self.data, 'y'):
            self.data.y = self.data.y.to(torch.float)


    @property
    def raw_dir(self):
        """Return the directory containing the raw dataset.

        The original implementation resolved the path two levels above the
        provided ``root`` directory.  As a consequence, the data loader looked
        for files such as ``dataset20/train`` in the repository root rather than
        inside the specified dataset folder (e.g. ``data/PowerGrid``).  This
        caused ``FileNotFoundError`` when the expected datasets were placed
        within ``root``.

        By using the supplied ``root`` directly we ensure that raw data is
        located relative to the dataset directory passed to the constructor and
        avoid missing file errors.
        """

        # ``self.root`` may be a relative path; make it absolute for
        # consistency with the rest of the PyG dataset API.
        return os.path.abspath(self.root)

    # @property
    def raw_file_names(self) -> List[str]:
        return [self.MARKER_FILE]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'valid.pt', 'test.pt']

    def download(self):
        marker_path = osp.join(self.raw_dir, self.MARKER_FILE)
        zip_path = osp.join(self.raw_dir, 'powergrid_data.zip')

        if not osp.exists(marker_path):
            os.makedirs(self.raw_dir, exist_ok=True)

            if not osp.exists(zip_path):
                print("Downloading data from Google Drive...")
                try:
                    gdown.download(self.URL, zip_path, quiet=False)
                except Exception as e:
                    raise RuntimeError(f"Failed to download file from Google Drive: {e}")
            else:
                print("Zip (extracted.marker) file already exists. Skipping download.")

            print("Extracting data...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_dir)
            except zipfile.BadZipFile:
                raise RuntimeError("Failed to extract zip file. The file might be corrupted.")

            os.remove(zip_path)
            print("Extraction completed.")

            with open(marker_path, 'w') as f:
                f.write('Extraction completed.')
            print("Download and extraction completed. Marker file created.")
        else:
            print("Datasets (extracted.marker file) already downloaded and extracted. Skipping download.")

    def process(self):
        for split in ['train', 'valid', 'test']:
            if split == 'test':
                path2= osp.join(self.raw_dir, self.test_dataset, split)
            else:
                path2 = osp.join(self.raw_dir, self.train_dataset, split)

            data_len, start_index, digits = get_length_of_dataset(path2)
            num_digits = '0' + str(len(str(digits)))
            data_list = []

            if split == 'test':
                file_path = osp.join(
                    self.raw_dir,
                    self.test_dataset,
                    'Netsci',
                    split,
                    'network_measures_final.csv',
                )
            else:
                file_path = osp.join(
                    self.raw_dir,
                    self.train_dataset,
                    'Netsci',
                    split,
                    'network_measures_final.csv',
                )

            # Load data
            csv_data = self.__read_NetSci_csv_file__(file_path)
            tensor_data = torch.tensor(csv_data, dtype=torch.float)

            len_sum = 0
            start_idx = 0
            for index in range(data_len):
                x, edge_index, edge_attr = self.__get_input__(
                    path2, num_digits, index + start_index
                )
                len_num = x.size(0)
                len_sum = len_sum + len_num

                Netsci_feature = tensor_data[start_idx:len_sum]
                start_idx = len_sum

                y = self.__get_label__(path2, num_digits, index + start_index)

                # Load Graphlets
                if split == 'test':
                    file_path_graphlets = osp.join(
                        self.raw_dir,
                        self.test_dataset,
                        'Graphlets',
                        split,
                    )
                else:
                    file_path_graphlets = osp.join(
                        self.raw_dir,
                        self.train_dataset,
                        'Graphlets',
                        split,
                    )

                Graphlets_feature = self.__get_graphlets__(
                    file_path_graphlets, num_digits, index + start_index
                )

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    Netsci_feature=Netsci_feature,
                    Graphlets_feature=Graphlets_feature,
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            print('save')
            # return self.collate(data_list)



    def __get_input__(self, path2, num_digits, index):
        id_index = format(index, num_digits)
        file_to_read = str(path2)+'/grid_data_'+str(id_index) + '.h5'
        hf = h5py.File(file_to_read, 'r')
        # read in sources/sinks
        dataset_P = hf.get('P')
        P = np.array(dataset_P) + 1
        # read in edge_index
        dataset_edge_index = hf.get('edge_index')
        edge_index = np.array(dataset_edge_index)-1
        # read in edge_attr
        dataset_edge_attr = hf.get('edge_attr')
        edge_attr = np.array(dataset_edge_attr)

        hf.close()

        # Cast node features to float to match model expectations
        x = (
            torch.tensor(P, dtype=torch.float)
            .unsqueeze(0)
            .t()
            .contiguous()
        )

        # x = torch.tensor(P).unsqueeze(0).t().to(torch.long).contiguous()
        edge_index = torch.tensor(edge_index).t().contiguous()

        from torch_geometric.utils import to_undirected, add_self_loops

        # 1) 变成无向（保证双向边都在）
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        # 2) （可选）加自环：GCNConv 会自动加，这里 LVConv 也可以不加
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_attr = torch.tensor(edge_attr).to(torch.long)
        return x, edge_index, edge_attr

    def __get_graphlets__(self, path2, num_digits, index):
        id_index = format(index, num_digits)
        file_to_read = str(path2)+'/4_size'+'/graphlets_'+str(id_index) + '.csv'
        csv_data = self.__read_graphlet_csv_file__(file_to_read)
        tensor_data = torch.tensor(csv_data, dtype=torch.float)

        return tensor_data


    def __get_label__(self, path2, num_digits, index):
        id_index = format(index, num_digits)
        if self.task == "snbs":
            file_to_read = str(path2)+'/snbs_'+str(id_index) + '.h5'
        elif self.task == "surv":
            file_to_read = str(path2)+'/surv_'+str(id_index) + '.h5'
        hf = h5py.File(file_to_read, 'r')
        if self.task == "snbs":
            dataset_target = hf.get('snbs')
        elif self.task == "surv":
            dataset_target = hf.get('surv')
        targets = np.array(dataset_target)
        hf.close()
        # Cast labels to float for compatibility with model outputs
        return torch.tensor(targets, dtype=torch.float)


    def __read_NetSci_csv_file__(self, file_path):
        '''here we delete some indexes which are not used in the paper'''
        with open(file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            data = []
            next(csvreader)  # ignore the header
            for row in csvreader:
                del row[49]
                del row[48]
                del row[47]
                del row[23]
                row = [float(element) for element in row]
                data.append(row)
            return data

    def __read_graphlet_csv_file__(self, file_path):
        with open(file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            data = []
            next(csvreader)
            for row in csvreader:
                row = [float(element) for element in row]
                data.append(row)
            return data
