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
    #print("overlap!!!")
    return True
  else: return False