'''
Currently working on this file.
Once complete it will allow the user to draw a number
which can be parsed into the model for evaluation
'''
import tkinter as tk
import Image, ImageDraw

win = tk.Tk()

win.geometry("700x300")

def draw_line(e):
   x, y = e.x, e.y
   if canvas.old_coords:
      x1, y1 = canvas.old_coords
      canvas.create_line(x, y, x1, y1, width=5)
   canvas.old_coords = x, y

canvas = tk.Canvas(win, width=700, height=300)
canvas.pack()
canvas.old_coords = None

image = Image.new("RGB",(700,300),(255,255,255))
draw = ImageDraw.Draw(image)
image.save("test1.jpg")
# Bind the left button the mouse.
win.bind('<ButtonPress-1>', draw_line)

win.mainloop()
