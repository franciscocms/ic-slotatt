import copy
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

def overflow(points):
  x0, y0, x1, y1 = points
  if x0 < 0. or x1 > 1. or y0 < 0. or y1 > 1.:
    #print("overflow!!!") 
    return True
  else: return False
    
def overlap_side(x0a, x1a, x0b, x1b):
  # works for x and y
  if x0a < x0b < x1a: return True
  elif x0b < x0a < x1b: return True
  elif x0a < x1b < x1a: return True
  elif x0b < x1a < x1b: return True
  else: return False

def overlap(points_a, points_b):
  # points should be for bounding box
  x0a, y0a, x1a, y1a = points_a
  x0b, y0b, x1b, y1b = points_b
  if overlap_side(x0a, x1a, x0b, x1b) and overlap_side(y0a, y1a, y0b, y1b): 
    return True
  else: return False

def color_to_rgb(color, transp):
  if color == "red": 
    if not transp: return (255, 0, 0)
    else: return (255, 0, 0, 125) 
  elif color == "green": 
    if not transp: return (0, 255, 0)
    else: return (0, 255, 0, 125)
  elif color == "blue": 
    if not transp: return (0, 0, 255)
    else: return (0, 0, 255, 125)
  elif color == "white": return (255, 255, 255)
  elif color == "black": return (0, 0, 0)

def img_to_tensor(image):
  img_transform = transforms.Compose([transforms.ToTensor()])
  return img_transform(image).unsqueeze(0)

def render(shape_obj, size_obj, color_obj, locx_obj, locy_obj, background, transparent_polygons=False):
  
  """
  - shape is in [ball, square]
  - color is in [red, green , blue]
  - size is in [small, medium, large] -> [10, 20, 40]
  - locX and locY are the mass centers
  """
  
  if background is None: background_code = (0, 0, 0)
  else: background_code = (int(background[0]*255),
                           int(background[1]*255),
                           int(background[2]*255))

  img = Image.new('RGB', (128, 128), background_code)
  draw = ImageDraw.Draw(img) if not transparent_polygons else ImageDraw.Draw(img, 'RGBA')

  for n in range(len(shape_obj)):
    shape, size, color = shape_obj[n], size_obj[n], color_obj[n]
    if shape != None: 
    
      locX, locY = locx_obj[n]*127, locy_obj[n]*127
      
      # for circles, s corresponds to radius
      if size == "small": s = 10
      elif size == "medium": s = 15
      elif size == "large": s = 20

      if shape == "square":
        draw.rectangle(
          [locX - s//2, locY - s//2, locX + s//2, locY + s//2], # (x0, y0, x1, y1)
          fill=color_to_rgb(color, transparent_polygons)
          )
      elif shape == "ball":
        draw.ellipse(
          [locX - s//2, locY - s//2, locX + s//2, locY + s//2],
          fill=color_to_rgb(color, transparent_polygons)
          )
  return img

def add_gaussian_mask(img, x_list, y_list, sigma=5.0):
    """
    Add a squared region with a Normal distribution centered at (x, y) to the image.

    Args:
    - image (torch.Tensor): Input image array.
    - x_list: x-coordinate of the center of the patch.
    - y_list: y-coordinate of the center of the patch.
    - sigma (float): Standard deviation of the Normal distribution.
    - size (int): Size of the patch (side length).

    Returns:
    - torch.Tensor: Image with the added Normal patch.
    """

    assert isinstance(x_list, list) and isinstance(y_list, list)
    assert len(x_list) == len(y_list)

    image = copy.deepcopy(img)
    returned_image = torch.zeros(image.shape)
    
    # Create a 2D grid representing the image
    grid_y, grid_x = torch.meshgrid(torch.arange(image.size(-2)), torch.arange(image.size(-1)))
    
    for i in range(len(x_list)):
        
        aux_image = copy.deepcopy(image)
        
        x = x_list[i]*(img.shape[-1]-1)
        y = y_list[i]*(img.shape[-1]-1)
    
        # Calculate the distance from each grid point to the center of the patch
        distances = torch.sqrt((grid_x - x) ** 2 + (grid_y - y) ** 2)
        
        # Calculate the Normal distribution
        patch = torch.exp(-distances.pow(2) / (2 * sigma**2))
        
        # Normalize the patch to have maximum value of 1
        patch /= patch.max()
        
        # Apply the patch to the image
        aux_image *= patch
        returned_image += aux_image
    
    return returned_image