import os
import time
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import numpy as np

def main(model_path, images_dir, confidence_threshold, num_classes, device='cpu', results_dir='results'):
    # Load the model state dictionary
    state_dict = torch.load(model_path, map_location=torch.device(device))

    # Create an instance of your model architecture
    model = fasterrcnn_resnet50_fpn_v2(num_classes=num_classes)
    
    model.load_state_dict(state_dict)

    # Move the model to the specified device
    model.to(device)

    model.eval()

    os.makedirs(results_dir, exist_ok=True)

    total_time = 0
    pbar = tqdm(os.listdir(images_dir))
    for i, image_name in enumerate(pbar):
        start_time = time.time()
        pbar.set_description(f'Processing image {i+1}/{len(os.listdir(images_dir))}: {image_name}')
        image_path = os.path.join(images_dir, image_name)

        image = Image.open(image_path)
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image).to(device)

        with torch.no_grad():
            prediction = model([image_tensor])
            boxes = prediction[0]['boxes'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()

            image_np = np.array(image)
            detections = 0
            for box, score in zip(boxes, scores):
                if score >= confidence_threshold:
                    detections += 1
                    x1, y1, x2, y2 = box
                    draw = ImageDraw.Draw(image)
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            pbar.set_postfix({'Detections': detections})
            tested_image_name = os.path.splitext(image_name)[0] + '_tested' + os.path.splitext(image_name)[1]
            tested_image_path = os.path.join(results_dir, tested_image_name)
            image.save(tested_image_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        pbar.set_postfix({'Detections': detections,'Time': f'{elapsed_time:.2f}s'})
    print(f'Total time taken: {total_time:.2f}s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faire de l\'inférence avec un modèle fasterrcnn_resnet50_fpn sur un répertoire d\'images.')
    parser.add_argument('model_path', type=str, help='Chemin du fichier du modèle')
    parser.add_argument('images_dir', type=str, help='Répertoire contenant les images à traiter')
    parser.add_argument('--confidence_threshold', type=float,default=0.5,help='Seuil de confiance en dessous duquel les détections ne sont pas prises en compte')
    parser.add_argument('--num_classes', type=int,default=2,help='Nombre de classes utilisées lors de l\'entraînement du modèle')
    parser.add_argument('--device', type=str,default='cpu',help='Device sur lequel exécuter le modèle (cpu ou cuda)')
    parser.add_argument('--results_dir', type=str,default='results',help='Répertoire dans lequel sauvegarder les images résultantes')
    args = parser.parse_args()
    # Set the PYTORCH_ENABLE_MPS_FALLBACK environment variable if the device is 'mps'
    if args.device == 'mps':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    print(f'PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}')
    main(args.model_path,args.images_dir,args.confidence_threshold,args.num_classes,args.device,args.results_dir)
