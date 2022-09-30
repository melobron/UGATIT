from torch.utils.data import Dataset
import cv2
from utils import *


class PairedDataset(Dataset):
    def __init__(self, domain1, domain2, train_size, train=True, transform=None):
        super(PairedDataset, self).__init__()

        img_dir = '../all_datasets/'
        domain1_dir = os.path.join(img_dir, domain1)
        domain2_dir = os.path.join(img_dir, domain2)
        if train:
            domain1_dir = os.path.join(domain1_dir, 'train')
            domain2_dir = os.path.join(domain2_dir, 'train')
        else:
            domain1_dir = os.path.join(domain1_dir, 'test')
            domain2_dir = os.path.join(domain2_dir, 'test')

        self.domain1_paths = sorted(make_dataset(domain1_dir))[:train_size]
        self.domain2_paths = sorted(make_dataset(domain2_dir))[:train_size]

        self.transform = transform

    def __getitem__(self, item):
        domain1_path = self.domain1_paths[item]
        domain2_path = self.domain2_paths[item]

        domain1_numpy = cv2.cvtColor(cv2.imread(domain1_path), cv2.COLOR_BGR2RGB)
        domain2_numpy = cv2.cvtColor(cv2.imread(domain2_path), cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=domain1_numpy, target=domain2_numpy)

        return {'domain1': transformed['image'], 'domain2': transformed['target']}

    def __len__(self):
        return len(self.domain1_paths)


if __name__ == "__main__":
    dataset = PairedDataset(domain1='Dog', domain2='Cat', train_size=3000)
    print(len(dataset))
