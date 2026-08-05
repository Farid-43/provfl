# reference: https://github.com/liuyugeng/ML-Doctor.git


import os
import torch
import pandas
import torchvision
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms

from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple

class CelebA(torch.utils.data.Dataset):
    base_folder = "celeba"

    def __init__(
            self,
            root: str,
            attr_list: str,
            prop_name: str,
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.attr_list = attr_list

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        self.prop_index = self.attr_names.index(prop_name)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # X = Image.open(os.path.join(self.root, self.base_folder, "img_celeba", self.filename[index]))
        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t, nums in zip(self.target_type, self.attr_list):
            if t == "attr":
                final_attr = 0
                for i in range(len(nums)):
                    final_attr += 2 ** i * self.attr[index][nums[i]]
                target.append(final_attr)
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))            

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None
        
        prop_label = self.attr[index][self.prop_index]
        # if prop_label:
        #     print('hello',self.filename[index], X.shape, target, prop_label)

        return X, target, prop_label

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


def get_celeba_dataset(dataset_name, attr, prop_name, root):
       
    if dataset_name.lower() == "celeba":
        if isinstance(attr, list):
            for a in attr:
                if a != "attr":
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))

                num_classes = [8, 4]
                # heavyMakeup MouthSlightlyOpen Smiling, Male Young
                attr_list = [[18, 21, 31], [20, 39]]
        else:
            if attr == "attr": # three types, hybrid
                num_classes = 8
                attr_list = [[18, 21, 31]]
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)), # reshape
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = CelebA(root=root, attr_list=attr_list, prop_name=prop_name, target_type=attr, transform=transform)
        input_channel = 3
    else:
        print('dataset error ...')

    return num_classes, dataset

# def split_data(data):
#     half = 32
#     x_a = data[:, :, :, 0:half]
#     x_b = data[:, :, :, half:64]
#     return x_a, x_b

# if __name__ == '__main__':
#     num_classes, dataset = get_model_dataset('celeba', attr='attr', prop_name='Eyeglasses', root='../data/')
#     celeba_len = 100000
#     train_celeba_len = int(celeba_len * 0.8)
#     test_celeba_len = int(celeba_len * 0.2)
#     train_dataset, test_dataset, _ = torch.utils.data.random_split(dataset, \
#                             [train_celeba_len, test_celeba_len, len(dataset)-train_celeba_len-test_celeba_len])

#     batch_size = 64
#     train_loader = torch.utils.data.DataLoader(
#             dataset=train_dataset,
#             batch_size=batch_size, shuffle=True,
#             # num_workers=args.workers
#         )
#     test_loader = torch.utils.data.DataLoader(
#             dataset=test_dataset,
#             batch_size=batch_size,
#             # num_workers=args.workers
#         )
    
#     for batch_idx, (input, target, prop) in enumerate(train_loader):
#         print(input.shape)
#         # print(target)
#         # print(prop)
#         x_a, x_b = split_data(input)
#         print(x_a.shape, x_b.shape)
#         break
        
# =================================== celebA attribute ====================================

# 0 5_o_Clock_Shadow
# 1 Arched_Eyebrows
# 2 Attractive
# 3 Bags_Under_Eyes
# 4 Bald
# 5 Bangs
# 6 Big_Lips
# 7 Big_Nose
# 8 Black_Hair
# 9 Blond_Hair
# 10 Blurry
# 11 Brown_Hair
# 12 Bushy_Eyebrows
# 13 Chubby
# 14 Double_Chin
# 15 Eyeglasses
# 16 Goatee
# 17 Gray_Hair
# 18 Heavy_Makeup
# 19 High_Cheekbones
# 20 Male
# 21 Mouth_Slightly_Open
# 22 Mustache
# 23 Narrow_Eyes
# 24 No_Beard
# 25 Oval_Face
# 26 Pale_Skin
# 27 Pointy_Nose
# 28 Receding_Hairline
# 29 Rosy_Cheeks
# 30 Sideburns
# 31 Smiling
# 32 Straight_Hair
# 33 Wavy_Hair
# 34 Wearing_Earrings
# 35 Wearing_Hat
# 36 Wearing_Lipstick
# 37 Wearing_Necklace
# 38 Wearing_Necktie
# 39 Young