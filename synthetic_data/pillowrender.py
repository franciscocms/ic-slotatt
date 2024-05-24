from PIL import Image, ImageDraw
import numpy as np

def color_to_rgb(color):
  if color == "red": return (255, 0, 0)
  elif color == "green": return (0, 255, 0)
  elif color == "blue": return (0, 0, 255)
  elif color == "white": return (255, 255, 255)
  elif color == "black": return (0, 0, 0)



def render(shape_obj, size_obj, color_obj, locx_obj, locy_obj, background_color):
  """
  shape is in [square, ball]
  color is in [red, green , blue]
  size is in [small, medium, large] -> [10, 20, 40]
  locX and locY are the mass centers

  """
  
  
  img = Image.new('RGB', (128, 128), color_to_rgb(background_color))
  draw = ImageDraw.Draw(img)

  for n in range(len(shape_obj)):
    shape, size, color, locX, locY = shape_obj[n], size_obj[n], color_obj[n], locx_obj[n]*127, locy_obj[n]*127

    # for circles, s corresponds to radius
    if size == "small": s = 10
    elif size == "medium": s = 15
    elif size == "large": s = 20

    if shape == "square":
      draw.rectangle(
        [locX - s//2, locY - s//2, locX + s//2, locY + s//2], # (x0, y0, x1, y1)
        fill=color_to_rgb(color))
    elif shape == "ball":
      draw.ellipse(
        [locX - s//2, locY - s//2, locX + s//2, locY + s//2],
        fill=color_to_rgb(color))
  #img.show()
  return img

def render_annot(locx_obj, locy_obj, background_color):
  """
  shape is in [square, ball]
  color is in [red, green , blue]
  size is in [small, medium, large] -> [10, 20, 40]
  locX and locY are the mass centers

  """
  
  img = Image.new('L', (128, 128), 0)
  draw = ImageDraw.Draw(img)

  for n in range(len(locx_obj)):
    locX, locY = locx_obj[n]*127, locy_obj[n]*127
    draw.point((locX, locY),fill=255)
  return img