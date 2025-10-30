import torch
from torchvision import transforms
from PIL import Image
from cnn import CNN

# Instantiate the model
model = CNN()

MODEL_STATE_FILENAME = 'trained_model.pt'
model.load_state_dict(torch.load(MODEL_STATE_FILENAME))
print(f'Model state loaded from: {MODEL_STATE_FILENAME}')

# load my data
img_PIL = Image.open("mydata/digit.png").convert('L')
img = transforms.ToTensor()(img_PIL)
img = 2 * (img - 0.5)
images_batch = torch.empty([1] + list(img.shape))
images_batch[0,0,:,:] = img

# run inference on my data
model.eval()
with torch.no_grad():
    outputs = model(images_batch)
    _, predicted = torch.max(outputs, 1)
    all_grades = outputs.tolist()[0]
    formatted_grades = [f'{num:.2f}' for num in all_grades]
    print(f'my data inference result - all grades:  ' + ' '.join(formatted_grades))
    print(f'my data inference result is: {predicted.item()}')




