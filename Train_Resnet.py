import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import transforms

from torchvision.ops import box_iou
from torchvision.io.image import read_image
from torchvision.transforms import ToPILImage
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import resize, pad
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

import os
import xml.etree.ElementTree as ET

from tqdm import tqdm

from PIL import Image, ImageDraw



#Fonction pour dessiner les tensors et les bounding box resized pour controler que tout est ok
def draw_bounding_boxes(image, target, i):
    # Create the Tensor directory if it doesn't exist
    if not os.path.exists('Tensor'):
        os.makedirs('Tensor')
    
    # Convert the image to a PIL Image
    to_pil = ToPILImage()
    pil_image = to_pil(image.mul(255).byte())
    
    # Draw the bounding boxes on the image
    draw = ImageDraw.Draw(pil_image)
    for box in target['boxes']:
        draw.rectangle(box.tolist(), outline='red', width=3)
    
    # Save the image with bounding boxes drawn in the Tensor directory
    pil_image.save(f'Tensor/image_{i}.png')


def collate_fn(batch):
    # Initialize lists for resized images and targets
    resized_images = []
    new_targets = []
    
    # Set the desired size
    desired_size = 512

    for i, (image, target) in enumerate(batch):
        # Get the width and height of the image
        w, h = image.shape[1:]
        if w > h :
            ratio = w / desired_size
            target_h = int(h / ratio)
            target_size = (desired_size, target_h)
            padding = [0, 0, desired_size - target_h, 0]
        else : 
            ratio = h / desired_size
            target_w = int(w / ratio)
            target_size = (target_w, desired_size)
            padding = [0, 0, 0, desired_size - target_w]
        # Resize the image to the target size

        resized_image = resize(image, target_size, antialias=True)
        padded_image = pad(resized_image, padding=padding)

        resized_images.append(padded_image)

        # Scale and shift the bounding boxes
        new_boxes = target['boxes'].clone()
        new_boxes[:, [0, 2]] /= ratio
        new_boxes[:, [1, 3]] /= ratio
        
        new_target = {
            'boxes': new_boxes,
            'labels': target['labels']
        }
        
        new_targets.append(new_target)

    return torch.stack(resized_images), new_targets


class CustomVOCDetection(Dataset):
    def __init__(self, root, classes):
        self.root = root
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.annotations_dir = os.path.join(root, 'Annotations')
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.label_map = {class_name: i + 1 for i, class_name in enumerate(classes)}

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        annotation_filename = os.path.splitext(image_filename)[0] + '.xml'
        annotation_path = os.path.join(self.annotations_dir, annotation_filename)
        
        # Load and process the XML annotations
        boxes, labels = self.load_annotations(annotation_path)
        
        # Convert the image to a tensor
        image = transforms.ToTensor()(image)
        
        labels = torch.tensor(labels)
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        return image, target

    def load_annotations(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            label = self.label_map[class_name]
            labels.append(label)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
        boxes = torch.tensor(boxes).float()
        return boxes, labels

    def __len__(self):
        return len(self.image_filenames)


def create_dataloaders(dataset, train_size, val_size, batch_size, workers):
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create a dataloader for the training set
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    
    # Create a dataloader for the validation set
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    pbar = tqdm(data_loader)
    for images, targets in pbar:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        pbar.set_description(f'Loss: {losses.item():.4f}')


def evaluate(model, data_loader, device):
    model.eval()
    aps = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes']
                pred_labels = output['labels']
                pred_scores = output['scores']
                true_boxes = target['boxes']
                true_labels = target['labels']
                ap = calculate_ap(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels)
                aps.append(ap)
    return sum(aps) / len(aps)


def calculate_ap(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels):
    iou_threshold = 0.5
    if true_labels.numel() == 0 or pred_labels.numel() == 0:
        return 0
    num_classes = max(true_labels.max(), pred_labels.max()) + 1
    aps = []
    for c in range(num_classes):
        # Get the predicted boxes and scores for class c
        class_pred_boxes = pred_boxes[pred_labels == c]
        class_pred_scores = pred_scores[pred_labels == c]
        
        # Get the true boxes for class c
        class_true_boxes = true_boxes[true_labels == c]
        
        # Sort the predicted boxes by their scores
        sorted_indices = torch.argsort(class_pred_scores, descending=True)
        class_pred_boxes = class_pred_boxes[sorted_indices]
        
        # Initialize the true positive and false positive arrays
        tp = torch.zeros(len(class_pred_boxes))
        fp = torch.zeros(len(class_pred_boxes))
        
        # Iterate over the predicted boxes
        for i, box in enumerate(class_pred_boxes):
            if len(class_true_boxes) == 0:
                # If there are no true boxes, mark this prediction as a false positive
                fp[i] = 1
            else:
                # Calculate the IoU between this box and all true boxes
                iou = box_iou(box.unsqueeze(0), class_true_boxes)[0]
                
                # Find the true box with the maximum IoU
                max_iou, max_index = iou.max(0)
                
                if max_iou >= iou_threshold:
                    # If the maximum IoU is above the threshold,
                    # mark this prediction as a true positive and remove the matched true box
                    tp[i] = 1
                    class_true_boxes = torch.cat([class_true_boxes[:max_index], class_true_boxes[max_index+1:]])
                else:
                    # Otherwise, mark this prediction as a false positive
                    fp[i] = 1
        
        # Calculate the precision and recall
        fp = fp.cumsum(0)
        tp = tp.cumsum(0)
        if len(true_boxes) == 0:
            recall = torch.zeros_like(tp)
        else:
            recall = tp / len(true_boxes)
        precision = tp / (tp + fp)
        
        # Calculate the average precision
        ap = 0
        for t in torch.linspace(0, 1, 11):
            if precision[recall >= t].nelement() == 0:
                p = 0
            else:
                p = precision[recall >= t].max().item()
            ap += p / 11
    
    return ap


def get_classes(annotations_dir):
    classes = set()
    for filename in os.listdir(annotations_dir):
        if filename.endswith('.xml'):
            tree = ET.parse(os.path.join(annotations_dir, filename))
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                classes.add(class_name)
    return sorted(list(classes))


def main(repertoire, device, batch_size, num_epochs, learning_rate, workers):
    if device == 'cuda' :
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Utiliser le troisieme GPU
        device = torch.device("cuda")
    annotations_dir = os.path.join(repertoire, 'Annotations')
    classes = get_classes(annotations_dir)
    dataset = CustomVOCDetection(root=repertoire, classes=classes)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_loader, val_loader = create_dataloaders(dataset, train_size, val_size, batch_size, workers)

    #weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    #model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    # On entraine le modele dans les poids par défaut sinon 
    # il va s'entrainer sur les 90 classes qu'il connait
    # On mets ici le nombre de classe à 2 car on a voiture
    # et on ajoute le background
    model = fasterrcnn_resnet50_fpn_v2( num_classes=2)
    model.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize the scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_mAP = 0
    counter = 0

    for epoch in range(num_epochs):
        # Train on one epoch
        train_one_epoch(model, optimizer, train_loader, device)
        
        # Step the scheduler
        scheduler.step()

        # Evaluate on validation set
        mAP = evaluate(model, val_loader, device)

        # Save the model if the mAP is the best so far
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), 'best_model.pth')
            counter = 0
        else:
            counter += 1

        # Print epoch results here
        print(f'Epoch: {epoch} - mAP: {mAP}')
        print(f'mAP: {mAP:.4f}')

        #Si au bout de 20 epoch ce n'est pas mieux on s'arrete
        if counter >= 20:
            break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a Faster R-CNN model on a custom dataset.')
    parser.add_argument('-f', '--repertoire', type=str,
                        help='The path to the dataset directory')
    parser.add_argument('-d', '--device', type=str,
                        help='The device to use for training')
    parser.add_argument('-b', '--batch', type=int,
                        help='The batch size')
    parser.add_argument('-e', '--epoch', type=int,
                        help='The number of epochs')
    parser.add_argument('-l', '--lr', type=float,
                        help='The learning rate')
    parser.add_argument('-w', '--workers', type=int,
                        help='The number of workers')
    args = parser.parse_args()

    main(args.repertoire,
         args.device,
         args.batch,
         args.epoch,
         args.lr,
         args.workers)
