from dependencies import *

def crop(image, new_shape):
    '''
    Input Params:
        1. Image (Type: Tensor):
            An input 4D tensor (Batchsize, channels, width, height).
        2. new_shape (type: tuple):
            A tuple containing a valid f 
        
        Given a Tensor and a shape new_shape crop function returns a new tensor of the required shape cropped from the center of the input
        tensor.
    
    Output Params:
        1. Cropped Image(Type: Tensor)
            Returns a tensor of shape new_shape cropped from center.
    '''
    if torch.is_tensor(image) == False:
        raise ValueError(
            'Crop function expects torch.tensor input, '
            f'got {type(image)}.'
        )   
    if len(new_shape) != len(image.shape):
        raise ValueError(
            'Crop function expects 4D input for image tensor (batch_size, channels, width, height), '
            f'got {type(image)}.'
        )
    for i in range(len(new_shape)):
        if(new_shape[i] > image.shape[i]):
            raise ValueError(
            'Crop function expects pixels in each dimension to be greater than or equal to new shape'
        )
    middle_height = torch.div(image.shape[2], 2, rounding_mode='floor')
    middle_width = torch.div(image.shape[3], 2, rounding_mode='floor')
    starting_height = middle_height - new_shape[2] / 2
    final_height = starting_height + new_shape[2]
    starting_width = middle_width - new_shape[3] / 2
    final_width = starting_width + new_shape[3]
    cropped_image = image[:, :, int(starting_height.item()):int(final_height.item()), int(starting_width.item()):int(final_width.item())]
    return cropped_image

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
        Input Params:
            1. Image_Tensor(Type: Tensor): 
                An Input 4D Tensor of shape (batch_size, channels, width, height).
            2. Num Images:
                Number of total Images in the tensor. Used to calculate grid size
                default = 25 (grid = 5 x 5). 
            3. Size:
                Size of each image in the given tensor of shape (channels, width, height).
                default = (1, 28, 28).
    
                Given a 4D Tensor containing a variety of images, size of images, number of images show_tensor_images returns a uniformly made grid of these images.

        Output Params:
            1. Uniformly made grid of images in the initial Tensor.
    '''
    if num_images <= 0:
        raise ValueError(
            'show_tensor_images function expects a Positive Integer of num_images in order to make a grid for the images.'
        )
    elif torch.is_tensor(image_tensor) == False:
        raise ValueError(
            'show_tensor_imagaes function expects torch.tensor input, '
            f'got {type(image_tensor)}.'
        )   
    elif len(image_tensor.shape) != 4:
        raise ValueError(
            'show_tensor_images function expects 4D input for image_tensor (batch_size, channels, width, height), '
            f'got {type(len(image_tensor.shape))}.'
        )
    elif len(size) != 3:
        raise ValueError(
            'show_tensor_images function expects 3D input for shape (channels, width, height)'
        )
    for i in range(1, len(image_tensor.shape)):
        if(size[i - 1] > image_tensor.shape[i]):
            raise ValueError(
            'show_tensor_images function expects pixels in each dimension to be greater than or equal to new shape'
        )
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    return image_grid.permute(1, 2, 0).squeeze()

def weights_init(m):
    '''
        Initializes weights to a small positive value and biases to 0.
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
