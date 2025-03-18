# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
#
# transform = transforms.Compose([
#
#         transforms.ToTensor(),
#         transforms.Resize((128, 128)),,
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#
# batch_size = 4
# data_dir = "data/gstrb"
# train_dataset = datasets.GTSRB(root=f"{data_dir}/train", split='train',target_transform=transform, download=True)
# val_dataset = datasets.GTSRB(root=f"{data_dir}/test",split='test', target_transform=transform, download=True)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
# print(keys(train_dataset))
