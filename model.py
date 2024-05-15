import os.path as osp
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

def process_single_image(model_path, test_img_path, output_path='images\image2.jpg'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    # Read the single image
    img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    print("hi")
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(output_path, output)

# Example usage:
model_path = 'RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
test_img_path = 'images\image.jpg'
output_path = 'images\image2.jpg'
process_single_image(model_path, test_img_path, output_path)